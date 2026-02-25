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
/// The reached node after traversal of `LRLRRLLR` branches (`L` for left, `R` for right) should be
/// represented as `0b0101100110000...0`.
/// Root is encoded as `0b10000...0` from an empty bitstring we don't need to traverse any branch
/// to reach a binary tree's root.
///
/// Here are some examples:
///
/// ```text
/// (root)    -> 0b10000000...0
/// L (left)  -> 0b01000000...0
/// R (right) -> 0b11000000...0
/// LL        -> 0b00100000...0
/// RLR       -> 0b10110000...0
/// LRL       -> 0b01010000...0
/// LRRLR     -> 0b01101100...0
/// ```
///
/// ## Multi-way tree
///
/// But we don't necessary need to encode a binary tree directly.
/// We can imagine some node to have `N` number of branches instead of two: right and left.
/// We encode `0 <= i < N` numbered branches by interpreting `i`'s binary representation as
/// bitstring for a binary tree traversal.
///
/// For example `N = 3`. Notice how right-most leaf node is unused:
///
/// ```text
///              root
///   root       /  \
///  / | \  =>  .    .
/// 0  1  2    / \  / \
///           0  1  2  -
/// ```
///
/// ## Order
///
/// Encoding allows to sort nodes in `left < parent < right` linear order.
/// If you only consider leaves of a tree then those are sorted in order `left < right`.
///
/// ## Used in
///
/// Primary purpose of `TreeNodeIndex` is to track order of parallel tasks of functions like
/// `par_join`, `par_slice`, and others (see `rustc_middle::sync`).
/// This is done in query cycle handling code to determine **intended** first task for a
/// single-threaded compiler front-end to execute even while multi-threaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TreeNodeIndex(u64);

impl TreeNodeIndex {
    /// Root node of a tree
    pub const fn root() -> Self {
        Self(0x80000000_00000000)
    }

    /// Append branch `i` out of `n` branches total to `TreeNodeIndex`'s traversal representation.
    ///
    /// This method reserves `ceil(log2(n))` bits within `TreeNodeIndex`'s integer encoded
    /// bitstring.
    pub fn branch(self, i: u64, n: u64) -> TreeNodeIndex {
        debug_assert!(i < n, "i = {i} should be less than n = {n}");
        // `branch_num != 0` per debug assertion above
        let bits = ceil_ilog2(n);

        let trailing_zeros = self.0.trailing_zeros();
        let allocated_shift = trailing_zeros.checked_sub(bits).expect(
            "TreeNodeIndex's free bits have been exhausted, make sure recursion is used carefully",
        );
        // Using wrapping operations for optimization, as edge cases are unreachable:
        // - `trailing_zeros < 64` as we are guaranteed at least one bit is set
        // - `allocated_shift == trailing_zeros - bits <= trailing_zeros < 64`
        TreeNodeIndex(
            self.0 & !u64::wrapping_shl(1, trailing_zeros)
                | u64::wrapping_shl(1, allocated_shift)
                | i.unbounded_shl(allocated_shift.wrapping_add(1)),
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
