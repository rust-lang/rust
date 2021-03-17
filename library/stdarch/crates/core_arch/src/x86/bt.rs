#[cfg(test)]
use stdarch_test::assert_instr;

/// Returns the bit in position `b` of the memory addressed by `p`.
#[inline]
#[cfg_attr(test, assert_instr(bt))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittest(p: *const i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        "bt [{p}], {b:e}",
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(readonly, nostack, pure)
    );
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then sets the bit to `1`.
#[inline]
#[cfg_attr(test, assert_instr(bts))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittestandset(p: *mut i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        "bts [{p}], {b:e}",
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(nostack)
    );
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then resets that bit to `0`.
#[inline]
#[cfg_attr(test, assert_instr(btr))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittestandreset(p: *mut i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        "btr [{p}], {b:e}",
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(nostack)
    );
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then inverts that bit.
#[inline]
#[cfg_attr(test, assert_instr(btc))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittestandcomplement(p: *mut i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        "btc [{p}], {b:e}",
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(nostack)
    );
    r
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;

    #[test]
    fn test_bittest() {
        unsafe {
            let a = 0b0101_0000i32;
            assert_eq!(_bittest(&a as _, 4), 1);
            assert_eq!(_bittest(&a as _, 5), 0);
        }
    }

    #[test]
    fn test_bittestandset() {
        unsafe {
            let mut a = 0b0101_0000i32;
            assert_eq!(_bittestandset(&mut a as _, 4), 1);
            assert_eq!(_bittestandset(&mut a as _, 4), 1);
            assert_eq!(_bittestandset(&mut a as _, 5), 0);
            assert_eq!(_bittestandset(&mut a as _, 5), 1);
        }
    }

    #[test]
    fn test_bittestandreset() {
        unsafe {
            let mut a = 0b0101_0000i32;
            assert_eq!(_bittestandreset(&mut a as _, 4), 1);
            assert_eq!(_bittestandreset(&mut a as _, 4), 0);
            assert_eq!(_bittestandreset(&mut a as _, 5), 0);
            assert_eq!(_bittestandreset(&mut a as _, 5), 0);
        }
    }

    #[test]
    fn test_bittestandcomplement() {
        unsafe {
            let mut a = 0b0101_0000i32;
            assert_eq!(_bittestandcomplement(&mut a as _, 4), 1);
            assert_eq!(_bittestandcomplement(&mut a as _, 4), 0);
            assert_eq!(_bittestandcomplement(&mut a as _, 4), 1);
            assert_eq!(_bittestandcomplement(&mut a as _, 5), 0);
            assert_eq!(_bittestandcomplement(&mut a as _, 5), 1);
        }
    }
}
