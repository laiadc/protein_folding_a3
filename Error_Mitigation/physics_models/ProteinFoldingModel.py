import torch
import numpy as np
import Error_Mitigation.utils as utils
from Error_Mitigation.physics_models.PhysicsModel import PhysicsModel

class ProteinFoldingModel(PhysicsModel):
    '''
    ElectronicStructureMolecule is a class of Hamiltonians that
    describe electronic structure models. It constructs the QubitHamiltonian
    from PySCFDriver
    *This class is also compatible with the PhysicsModel class
    which is used during NN training and VMC by
    > Defining the get_hamiltonian_lines(samples) function by
        Converting the qubit Hamitlonian into a sparse matrix which
        is used in get_hamiltonian_lines & PhysicsModel functions
    '''
    def __init__(self,qubit_op):
        '''
        Parameters
            -----------
            
        '''
        super().__init__() #Initialize PhysicsModel
        #Qubit Hamiltonian is input into
        #Qiskit VQE functions
        self.qubit_hamiltonian = qubit_op
        #Save Sparse Hamiltonian instead of 2^n * 2^n
        #matrix --> used in get_hamiltonian_lines

        self.sp_hamiltonian_csr = self.qubit_hamiltonian.to_spmatrix()
        sp_ham_coo = self.sp_hamiltonian_csr.tocoo()
        #pytorch sparse matrix form to be compatible with .to(device_gpu)
        self.sp_hamiltonian = torch.sparse.LongTensor(torch.LongTensor([
                    sp_ham_coo.row.tolist(), sp_ham_coo.col.tolist()]),
                    torch.Tensor(sp_ham_coo.data.real.astype(np.float32)))
        #Define the number of sites for the PhysicsModel
        self.num_qubits = qubit_op.num_qubits
        self.num_sites = self.num_qubits

    def get_hamiltonian_lines(self, samples, **kwargs):
        '''
        Takes a batch of samples and outputs the contributing lines of the
        Hamiltonian and the indicies (states). We do this using the sparse matrix
        obtained from the QubitHamiltonian
        Parameters
        ----------------
        samples: Tensor output of utils.long_to_bits(samples)

        Returns
        ----------------
        Contributing_states_pd Torch.Tensor
            Lists the computational basis states with non-zero entries in Hamiltonian
            for each sampled state in samples.
        Contributing_data_pd: Complex tensor
            Lists the non-zerp Hamiltonian row elements for each sampled state in samples.
        '''
        samples_long = utils.bits_to_long(samples)

        self.sp_hamiltonian = self.sp_hamiltonian.to(samples.device)
        #Separate the sparse matrix for the sampled in samples
        contributing_elements = [ self.sp_hamiltonian[s_long] for s_long in samples_long]
        # Save the indicides (correspond to states) from non-zero elements of H[sample]
        contributing_states = [ c_elements._indices()[0].to( dtype=torch.int64 )
                                        for c_elements in contributing_elements]
        # print ("contributing states = ", contributing_states)
        # Save the non-zero elements of H[sample]
        contributing_data = [ utils.Complex( c_elements._values() ).to(samples.device).to(dtype = torch.float)
                                            for c_elements in contributing_elements]

        # Need to "pad" the lists so that the tensor has a defined shape
        # and each array is the same length
        # 1. Given the samples, find the the max number of contributing states (non-zero Hamiltonian elements)
        max_contributing_states = max( [ len(c) for c in contributing_states ] )
        # 3. 'Pad' the shorter arrays with the 0 c.b state
        contributing_states_long_pd = torch.stack(
            [ torch.cat( [ s , torch.zeros(max_contributing_states - len(s), dtype=torch.int64).to(samples.device)]
            , 0) for s in contributing_states] )
        # 3. 'Pad' the shorter data arrays with zeros so the 'padded' 0 c.b states
        # do not change the Hamiltonian dynamics
        contributing_data_pd = utils.Complex.stack([utils.Complex.cat(
            [ data , utils.Complex( torch.zeros( max_contributing_states - len(data),
                            dtype=torch.float).to(samples.device)) ])
                            for data in contributing_data])

        #Convert the sampled states from long form (0,1,2,3...2^n) to bit strings [0010...N]
        contributing_states_pd = utils.long_to_bits( contributing_states_long_pd, self.num_sites )

        return contributing_states_pd, contributing_data_pd

    

    