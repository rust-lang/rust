//! Read-shared Move instructions

#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.movrsqi"]
    fn movrsqi(src: *const i8) -> i8;
    #[link_name = "llvm.x86.movrshi"]
    fn movrshi(src: *const i16) -> i16;
    #[link_name = "llvm.x86.movrssi"]
    fn movrssi(src: *const i32) -> i32;
    #[link_name = "llvm.x86.movrsdi"]
    fn movrsdi(src: *const i64) -> i64;
}

/// Moves a byte from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i8(src: *const i8) -> i8 {
    movrsqi(src)
}

/// Moves a 16-bit word from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i16(src: *const i16) -> i16 {
    movrshi(src)
}

/// Moves a 32-bit doubleword from the source to the destination, with an indication that the source
/// memory location is likely to become read-shared by multiple processors, i.e., read in the future
/// by at least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i32(src: *const i32) -> i32 {
    movrssi(src)
}

/// Moves a 64-bit quadword from the source to the destination, with an indication that the source
/// memory location is likely to become read-shared by multiple processors, i.e., read in the future
/// by at least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i64(src: *const i64) -> i64 {
    movrsdi(src)
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use super::*;

    #[simd_test(enable = "movrs")]
    fn test_movrs_i8() {
        let x: i8 = 42;
        let y = unsafe { _movrs_i8(&x) };
        assert_eq!(x, y);
    }

    #[simd_test(enable = "movrs")]
    fn test_movrs_i16() {
        let x: i16 = 42;
        let y = unsafe { _movrs_i16(&x) };
        assert_eq!(x, y);
    }

    #[simd_test(enable = "movrs")]
    fn test_movrs_i32() {
        let x: i32 = 42;
        let y = unsafe { _movrs_i32(&x) };
        assert_eq!(x, y);
    }

    #[simd_test(enable = "movrs")]
    fn test_movrs_i64() {
        let x: i64 = 42;
        let y = unsafe { _movrs_i64(&x) };
        assert_eq!(x, y);
    }
}
