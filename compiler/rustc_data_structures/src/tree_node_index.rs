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
pub struct TreeNodeIndex(u64);

impl TreeNodeIndex {
    pub const fn root() -> Self {
        Self(0x80000000_00000000)
    }

    /// Append tree branch no. `branch_idx` reserving `bits` bits.
    fn try_bits_branch(self, branch_idx: u64, bits: u32) -> Option<Self> {
        let trailing_zeros = self.0.trailing_zeros();
        let allocated_shift = trailing_zeros.checked_sub(bits)?;
        // Using wrapping operations for optimization, as edge cases are unreachable:
        // - `trailing_zeros < 64` as we are guaranteed at least one bit is set
        // - `allocated_shift == trailing_zeros - bits <= trailing_zeros < 64`
        Some(TreeNodeIndex(
            self.0 & !u64::wrapping_shl(1, trailing_zeros)
                | u64::wrapping_shl(1, allocated_shift)
                | branch_idx.unbounded_shl(allocated_shift.wrapping_add(1)),
        ))
    }

    /// Append tree branch no. `branch_idx` reserving `ceil(log2(branch_num))` bits.
    pub fn branch(self, branch_idx: u64, branch_num: u64) -> TreeNodeIndex {
        debug_assert!(
            branch_idx < branch_num,
            "branch_idx = {branch_idx} should be less than branch_num = {branch_num}"
        );
        // `branch_num != 0` per debug assertion above
        let bits = ceil_ilog2(branch_num);
        self.try_bits_branch(branch_idx, bits).expect(
            "TreeNodeIndex's free bits have been exhausted, make sure recursion is used carefully",
        )
    }
}

#[inline]
fn ceil_ilog2(branch_num: u64) -> u32 {
    // Using `wrapping_sub` for optimization, consider `log(0)` to be undefined
    // `floor(log2(n - 1)) + 1 == ceil(log2(n))`
    branch_num.wrapping_sub(1).checked_ilog2().map_or(0, |b| b.wrapping_add(1))
}

#[cfg(test)]
mod tests;
