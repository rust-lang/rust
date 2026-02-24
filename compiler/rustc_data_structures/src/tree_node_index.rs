use std::error::Error;
use std::fmt::Display;

/// Ordered index for dynamic trees
///
/// ## Encoding
///
/// You can index any node of a binary tree with a variable-length bitstring.
/// Each bit should represent a branch/edge to traverse up from root to down to indexed node.
/// To encode a variable-length bitstring we use bits of u64 ordered from highest to lowest.
/// Right after the encoded sequence of bits we set one bit high to recover sequence's length:
///
/// ```text
/// 0bXXXXXXX100000000...0
/// ```
///
/// Node reach after traversal of `LRLRRLLR` branches should be represented as `0b0101100110000...0`.
/// Root is obviously encoded as `0b10000...0`.
///
/// ## Order
///
/// Encoding allows to sort nodes in left < parent < right linear order.
/// If you only consider leaves of a tree then those are sorted in order left < right.
///
/// ## Used in
///
/// Primary purpose of `TreeNodeIndex` is to track order of parallel tasks of functions like `join`
/// or `scope` (see `rustc_middle::sync`).
/// This is done in query cycle handling code to determine **intended** first task for a single-
/// threaded compiler front-end to execute even while multi-threaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TreeNodeIndex(pub u64);

impl TreeNodeIndex {
    pub const fn root() -> Self {
        Self(0x80000000_00000000)
    }

    /// Append tree branch no. `branch_idx` reserving `bits` bits.
    fn try_bits_branch(self, branch_idx: u64, bits: u32) -> Result<Self, BranchingError> {
        let trailing_zeros = self.0.trailing_zeros();
        let allocated_shift = trailing_zeros.checked_sub(bits).ok_or(BranchingError(()))?;
        Ok(TreeNodeIndex(
            self.0 & !(1 << trailing_zeros)
                | (1 << allocated_shift)
                | (branch_idx << (allocated_shift + 1)),
        ))
    }

    /// Append tree branch no. `branch_idx` reserving `ceil(log2(branch_num))` bits.
    pub fn branch(self, branch_idx: u64, branch_num: u64) -> TreeNodeIndex {
        debug_assert!(
            branch_idx < branch_num,
            "branch_idx = {branch_idx} should be less than branch_num = {branch_num}"
        );
        // floor(log2(n - 1)) + 1 == ceil(log2(n))
        self.try_bits_branch(branch_idx, (branch_num - 1).checked_ilog2().map_or(0, |b| b + 1))
            .unwrap()
    }

    pub fn try_concat(self, then: Self) -> Result<Self, BranchingError> {
        let trailing_zeros = then.0.trailing_zeros();
        let branch_num = then.0.wrapping_shr(trailing_zeros + 1);
        let bits = u64::BITS - trailing_zeros;
        self.try_bits_branch(branch_num, bits)
    }
}

/// Error for exhausting free bits
#[derive(Debug)]
pub struct BranchingError(());

impl Error for BranchingError {}

impl Display for BranchingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        "TreeNodeIndex's free bits have been exhausted, make sure recursion is used carefully"
            .fmt(f)
    }
}

impl Default for TreeNodeIndex {
    fn default() -> Self {
        TreeNodeIndex::root()
    }
}
