#[cfg(test)]
use stdarch_test::assert_instr;

/// Returns the bit in position `b` of the memory addressed by `p`.
#[inline]
#[cfg_attr(test, assert_instr(bt))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittest64(p: *const i64, b: i64) -> u8 {
    let r: u8;
    llvm_asm!("btq $2, $1\n\tsetc ${0:b}"
              : "=r"(r)
              : "*m"(p), "r"(b)
              : "cc", "memory");
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then sets the bit to `1`.
#[inline]
#[cfg_attr(test, assert_instr(bts))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittestandset64(p: *mut i64, b: i64) -> u8 {
    let r: u8;
    llvm_asm!("btsq $2, $1\n\tsetc ${0:b}"
              : "=r"(r), "+*m"(p)
              : "r"(b)
              : "cc", "memory");
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then resets that bit to `0`.
#[inline]
#[cfg_attr(test, assert_instr(btr))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittestandreset64(p: *mut i64, b: i64) -> u8 {
    let r: u8;
    llvm_asm!("btrq $2, $1\n\tsetc ${0:b}"
              : "=r"(r), "+*m"(p)
              : "r"(b)
              : "cc", "memory");
    r
}

/// Returns the bit in position `b` of the memory addressed by `p`, then inverts that bit.
#[inline]
#[cfg_attr(test, assert_instr(btc))]
#[unstable(feature = "simd_x86_bittest", issue = "59414")]
pub unsafe fn _bittestandcomplement64(p: *mut i64, b: i64) -> u8 {
    let r: u8;
    llvm_asm!("btcq $2, $1\n\tsetc ${0:b}"
              : "=r"(r), "+*m"(p)
              : "r"(b)
              : "cc", "memory");
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
