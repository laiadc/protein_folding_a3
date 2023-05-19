from qiskit_research.protein_folding.interactions.random_interaction import (
    RandomInteraction,
)
from qiskit_research.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_research.protein_folding.peptide.peptide import Peptide
from qiskit_research.protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)

from qiskit_research.protein_folding.penalty_parameters import PenaltyParameters

from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.providers.fake_provider import FakeManila

from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Options
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.primitives import BackendSampler

algorithm_globals.random_seed = 23

import itertools
from qiskit.quantum_info import SparsePauliOp
import re
import numpy as np
import Error_Mitigation.data_utils as data_utils
from qiskit import QuantumCircuit
import numpy as np

def get_ground_state(qubit_op):
    '''
    Returns the bit-string with the minimum energy, by trying all the possible basis states.
    This function is used to check the output of the quantum VQE.

    Args:
        qubit_op: Hamiltonian to minimize
    Output:
        idx: Indices ordered with increasing energy
        energy: Energy of the basis states corresponding to indices idx
    '''
    all_results = []

    for i in range(2**qubit_op.num_qubits):
        estimator = Estimator()
        # Create quantum circuit that produces initial state 
        initial_state = np.zeros(2**qubit_op.num_qubits)
        initial_state[i] = 1
        qc = QuantumCircuit(qubit_op.num_qubits)
        qc.initialize(initial_state)
        job = estimator.run(qc, qubit_op)
        result = job.result()
        all_results.append(result.values[0])

    idx = np.argsort(all_results)
    return idx, np.array(all_results)[idx]


def run_VQE(main_chain='APRLRFY', side_chains=None, interaction='MJ',penalty_back=10,penalty_chiral=10,penalty_1=10, 
            fake_backend=None, noise_model='depolarizing',p=0.01, resilience_level=0, aggregation=0.1):
    ##################################
    # 1. DEFINE PROTEIN CHAIN
    ##################################
    if side_chains is None:
        side_chains = [""]*len(main_chain)
    penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)
    if interaction == 'MJ':
        interaction = MiyazawaJerniganInteraction()
    elif interaction=='random':
        interaction = RandomInteraction()
    peptide = Peptide(main_chain, side_chains)

    protein_folding_problem = ProteinFoldingProblem(peptide, interaction, penalty_terms)
    qubit_op = protein_folding_problem.qubit_op()
    num_qubits = qubit_op.num_qubits

    ####################################
    # 2. DEFINE NOISE MODEL
    ####################################
    if fake_backend is None:
        if noise_model=='depolarizing':
            # Depolarizing quantum errors
            error_1 = noise.depolarizing_error(p, 1)
            error_2 = noise.depolarizing_error(p, 2)
        elif noise_model=='amplitude_damping':
            error_1 = noise.amplitude_damping_error(p)
            error_2 = noise.amplitude_damping_error(p)
            error_2 = error_1.tensor(error_2)
        elif noise_model=='phase_damping':
            error_1 = noise.phase_damping_error(p, 1)
            error_2 = noise.phase_damping_error(p, 2)
            error_2 = error_1.tensor(error_2)
        else:
            raise ValueError("Noise model not implemented ", noise_model)
        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['x','h','ry','rz','rx','u'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

        # Get basis gates from noise model
        basis_gates = noise_model.basis_gates
        simulator = {'noise_model': noise_model, 'seed_simulator':1234, 'basis_gates':basis_gates}

        # 1. Initialize account
        service = QiskitRuntimeService(channel="ibm_quantum")

        # 2. Specify options, such as enabling error mitigation
        options = Options(resilience_level=resilience_level, simulator=simulator)

        # 3. Select a backend.
        backend = service.backend("ibmq_qasm_simulator")
        sampler = Sampler(session=backend, options=options)
    else:
        backend = fake_backend
        options = Options(resilience_level=resilience_level)
        sampler = BackendSampler(backend = backend)#, options=options)
        
    ####################################
    # 3. RUN VQE
    ####################################
    optimizer = COBYLA(maxiter=50)

    # set variational ansatz
    ansatz = RealAmplitudes(reps=1)

    counts_noise = []
    values_noise = []
    params_noise = []


    def store_intermediate_result(eval_count, parameters, mean, std):
        counts_noise.append(eval_count)
        values_noise.append(mean)
        params_noise.append(parameters)


    # initialize VQE using CVaR with alpha = 0.1
    vqe = SamplingVQE(
        sampler,
        ansatz=ansatz,
        optimizer=optimizer,
        aggregation=aggregation,
        callback=store_intermediate_result,
    )

    raw_result_noise = vqe.compute_minimum_eigenvalue(qubit_op)

    return raw_result_noise, counts_noise, values_noise, params_noise, ansatz, vqe, noise_model, sampler, qubit_op


import torch
def decimal_to_bin(n, num_qubits=9):
    b = bin(n).replace("0b", "") 
    if (num_qubits - len(b))>0:
       b = "0"* (num_qubits - len(b)) + b
    return b


def get_measurement_dict(ansatz, params_noise, sampler, num_samples_per_basis=3000):
    num_qubits = ansatz.num_qubits
    num_pairs = int(num_qubits*(num_qubits-1)/2)
    bases_z   = [num_qubits * ['Z'] for _ in range(1)]
    bases_zx  = [num_qubits * ['Z'] for _ in range(num_qubits)]
    bases_zxx = [num_qubits * ['Z'] for _ in range(num_pairs)]
    for i in range(num_qubits): # One X measurement
        bases_zx[i][i] = 'X'
    row = 0
    for n in range(num_qubits): # Two X Measurements
        for m in range(n+1,num_qubits):
            bases_zxx[row][n] = 'X'
            bases_zxx[row][m] = 'X'
            row += 1
    bases_list = list(itertools.chain(bases_z, bases_zx, bases_zxx))
    bases = []
    for b in bases_list:
        obs_name = ''.join(b)
        observable = SparsePauliOp(obs_name)
        bases.append(obs_name)


    qc = ansatz.bind_parameters(params_noise[-1])
    qc.remove_final_measurements()
    circuits = []
    for b in bases:
        q_aux = qc.copy()
        pos = [match.start() for match in re.finditer('X', b[::-1])]
        for p in pos:
            q_aux.h(p)
        q_aux.measure_all()
        circuits.append(q_aux)

    job = sampler.run(circuits)
    result = job.result()
    dists = result.quasi_dists


    measurement_dict= {}
    # For loop to cover all bases
    for i in range(len(dists)):
        d = dists[i]
        base = bases[i]
        indices = list(d.keys())
        binary_strings = [decimal_to_bin(n, num_qubits) for n in indices]
        counts = np.array(list(d.values()))*result.metadata[0]['shots']

        # For loop to create the arrays
        measurement_array= []
        for j in range(len(counts)):
            c = counts[j]
            binary = binary_strings[j]
            for k in range(int(c)):
                measurement_array.append(list(map(int, list(binary))))
        np.random.shuffle(measurement_array) 
        measurement_dict[base] = torch.tensor(measurement_array, dtype=torch.uint8)



    bases = data_utils.str_to_list(measurement_dict.keys())
    print ("Which measurements were taken", measurement_dict.keys())
    print ('bases = ', bases)
    measurement_data = data_utils.circuit_samples_to_MeasurementsDataset(
                                        measurement_dict,
                                        bases,
                                        num_samples_per_basis)
    return measurement_data, bases, dists
