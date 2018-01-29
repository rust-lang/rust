#[cfg(test)]
use stdsimd_test::assert_instr;

/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
#[inline(always)]
#[target_feature(enable = "lzcnt")]
#[cfg_attr(test, assert_instr(lzcnt))]
pub unsafe fn _lzcnt_u64(x: u64) -> u64 {
    x.leading_zeros() as u64
}

/// Counts the bits that are set.
#[inline(always)]
#[target_feature(enable = "popcnt")]
#[cfg_attr(test, assert_instr(popcnt))]
pub unsafe fn _popcnt64(x: i64) -> i32 {
    x.count_ones() as i32
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use x86::*;

    #[simd_test = "lzcnt"]
    unsafe fn test_lzcnt_u64() {
        assert_eq!(_lzcnt_u64(0b0101_1010), 57);
    }

    #[simd_test = "popcnt"]
    unsafe fn test_popcnt64() {
        assert_eq!(_popcnt64(0b0101_1010), 4);
    }
}
