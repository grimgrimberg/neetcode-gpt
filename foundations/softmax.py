import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # 1. Subtract the max value from all elements for numerical stability
        stable_z = z - np.max(z)
        
        # 2. Exponentiate the stabilized logits vectorially
        exp_z = np.exp(stable_z)
        
        # 3. Divide each exponent by the total sum and round to 4 decimal places
        softmax_probs = exp_z / np.sum(exp_z)
        
        return np.round(softmax_probs, 4)
