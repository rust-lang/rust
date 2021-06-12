#[cfg(test)]
use stdarch_test::assert_instr;

// x32 wants to use a 32-bit address size, but asm! defaults to using the full
// register name (e.g. rax). We have to explicitly override the placeholder to
// use the 32-bit register name in that case.
#[cfg(target_pointer_width = "32")]
macro_rules! bt {
    ($inst:expr) => {
        concat!($inst, " {b}, ({p:e})")
    };
}
#[cfg(target_pointer_width = "64")]
macro_rules! bt {
    ($inst:expr) => {
        concat!($inst, " {b}, ({p})")
    };
}

/// Returns the bit in position `b` of the memory addressed by `p`.
#[inline]
#[cfg_attr(test, assert_instr(bt))]
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub unsafe fn _bittest64(p: *const i64, b: i64) -> u8 {
    let r: u8;
    asm!(
        bt!("btq"),
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(readonly, nostack, pure, att_syntax)
    );
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then sets the bit to `1`.
#[inline]
#[cfg_attr(test, assert_instr(bts))]
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub unsafe fn _bittestandset64(p: *mut i64, b: i64) -> u8 {
    let r: u8;
    asm!(
        bt!("btsq"),
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(nostack, att_syntax)
    );
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then resets that bit to `0`.
#[inline]
#[cfg_attr(test, assert_instr(btr))]
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub unsafe fn _bittestandreset64(p: *mut i64, b: i64) -> u8 {
    let r: u8;
    asm!(
        bt!("btrq"),
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(nostack, att_syntax)
    );
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then inverts that bit.
#[inline]
#[cfg_attr(test, assert_instr(btc))]
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub unsafe fn _bittestandcomplement64(p: *mut i64, b: i64) -> u8 {
    let r: u8;
    asm!(
        bt!("btcq"),
        "setc {r}",
        p = in(reg) p,
        b = in(reg) b,
        r = out(reg_byte) r,
        options(nostack, att_syntax)
    );
    r
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86_64::*;

    #[test]
    fn test_bittest64() {
        unsafe {
            let a = 0b0101_0000i64;
            assert_eq!(_bittest64(&a as _, 4), 1);
            assert_eq!(_bittest64(&a as _, 5), 0);
        }
    }

    #[test]
    fn test_bittestandset64() {
        unsafe {
            let mut a = 0b0101_0000i64;
            assert_eq!(_bittestandset64(&mut a as _, 4), 1);
            assert_eq!(_bittestandset64(&mut a as _, 4), 1);
            assert_eq!(_bittestandset64(&mut a as _, 5), 0);
            assert_eq!(_bittestandset64(&mut a as _, 5), 1);
        }
    }

    #[test]
    fn test_bittestandreset64() {
        unsafe {
            let mut a = 0b0101_0000i64;
            assert_eq!(_bittestandreset64(&mut a as _, 4), 1);
            assert_eq!(_bittestandreset64(&mut a as _, 4), 0);
            assert_eq!(_bittestandreset64(&mut a as _, 5), 0);
            assert_eq!(_bittestandreset64(&mut a as _, 5), 0);
        }
    }

    #[test]
    fn test_bittestandcomplement64() {
        unsafe {
            let mut a = 0b0101_0000i64;
            assert_eq!(_bittestandcomplement64(&mut a as _, 4), 1);
            assert_eq!(_bittestandcomplement64(&mut a as _, 4), 0);
            assert_eq!(_bittestandcomplement64(&mut a as _, 4), 1);
            assert_eq!(_bittestandcomplement64(&mut a as _, 5), 0);
            assert_eq!(_bittestandcomplement64(&mut a as _, 5), 1);
        }
    }
}
