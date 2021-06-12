#[cfg(test)]
use stdarch_test::assert_instr;

// x32 wants to use a 32-bit address size, but asm! defaults to using the full
// register name (e.g. rax). We have to explicitly override the placeholder to
// use the 32-bit register name in that case.
#[cfg(target_pointer_width = "32")]
macro_rules! bt {
    ($inst:expr) => {
        concat!($inst, " {b:e}, ({p:e})")
    };
}
#[cfg(target_pointer_width = "64")]
macro_rules! bt {
    ($inst:expr) => {
        concat!($inst, " {b:e}, ({p})")
    };
}

/// Returns the bit in position `b` of the memory addressed by `p`.
#[inline]
#[cfg_attr(test, assert_instr(bt))]
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub unsafe fn _bittest(p: *const i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        bt!("btl"),
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
pub unsafe fn _bittestandset(p: *mut i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        bt!("btsl"),
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
pub unsafe fn _bittestandreset(p: *mut i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        bt!("btrl"),
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
pub unsafe fn _bittestandcomplement(p: *mut i32, b: i32) -> u8 {
    let r: u8;
    asm!(
        bt!("btcl"),
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
