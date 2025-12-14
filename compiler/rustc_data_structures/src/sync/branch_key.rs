use std::cmp;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BranchKey(u128);

impl BranchKey {
    pub const fn root() -> Self {
        Self(0x80000000_00000000_00000000_00000000)
    }

    fn bits_branch(self, branch_num: u128, bits: u32) -> Result<Self, BranchNestingError> {
        let trailing_zeros = self.0.trailing_zeros();
        let allocated_shift = trailing_zeros.checked_sub(bits).ok_or(BranchNestingError(()))?;
        Ok(BranchKey(
            self.0 & !(1 << trailing_zeros)
                | (1 << allocated_shift)
                | (branch_num << (allocated_shift + 1)),
        ))
    }

    pub fn branch(self, branch_num: u128, branch_space: u128) -> BranchKey {
        debug_assert!(
            branch_num < branch_space,
            "branch_num = {branch_num} should be less than branch_space = {branch_space}"
        );
        // floor(log2(n - 1)) + 1 == ceil(log2(n))
        self.bits_branch(branch_num, (branch_space - 1).checked_ilog2().map_or(0, |b| b + 1))
            .expect("query branch space is exhausted")
    }

    pub fn disjoint_cmp(self, other: Self) -> cmp::Ordering {
        self.0.cmp(&other.0)
    }

    pub fn nest(self, then: Self) -> Result<Self, BranchNestingError> {
        let trailing_zeros = then.0.trailing_zeros();
        let branch_num = then.0.wrapping_shr(trailing_zeros + 1);
        let bits = u128::BITS - trailing_zeros;
        self.bits_branch(branch_num, bits)
    }
}

#[derive(Debug)]
pub struct BranchNestingError(());

impl Default for BranchKey {
    fn default() -> Self {
        BranchKey::root()
    }
}
