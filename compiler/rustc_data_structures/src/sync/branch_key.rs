use std::cmp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BranchKey(u128);

impl BranchKey {
    pub fn root() -> Self {
        Self(0x80000000_00000000_00000000_00000000)
    }

    pub fn bit_branch(self) -> [Self; 2] {
        let trailing_zeroes = self.0.trailing_zeros();
        assert!(trailing_zeroes >= 1, "query branch space is exhausted");
        [BranchKey(self.0 ^ (0b11 << (trailing_zeroes - 1))), BranchKey(self.0 ^ (0b01 << (trailing_zeroes - 1)))]
    }

    pub fn bits_branch_iter(self, bits: u32) -> impl Iterator<Item = Self> {
        let trailing_zeroes = self.0.trailing_zeros();
        let allocated_shift = trailing_zeroes
            .checked_sub(bits)
            .unwrap_or_else(|| panic!("query branch space is exhausted to fit {bits} bits"));
        let step = 1 << (allocated_shift + 1);
        let zero = self.0 & !(1 << trailing_zeroes) | (1 << allocated_shift);
        (0..1 << bits).map(move |n| {
            BranchKey(zero + step * n)
        })
    }

    pub fn n_branch_iter(self, n: u32) -> impl Iterator<Item = Self> {
        let iter = self.bits_branch_iter(n.saturating_sub(1).checked_ilog2().map_or(0, |b| b + 1));
        iter.take(n.try_into().unwrap())
    }

    pub fn raw_cmp(self, other: Self) -> cmp::Ordering {
        self.0.cmp(&other.0)
    }
}


