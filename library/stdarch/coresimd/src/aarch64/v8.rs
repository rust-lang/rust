//! ARMv8 intrinsics.
//!
//! The reference is [ARMv8-A Reference Manual][armv8].
//!
//! [armv8]: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.
//! ddi0487a.k_10775/index.html

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Reverse the order of the bytes.
#[inline(always)]
#[cfg_attr(test, assert_instr(rev))]
pub unsafe fn _rev_u64(x: u64) -> u64 {
    x.swap_bytes() as u64
}

/// Count Leading Zeros.
#[inline(always)]
#[cfg_attr(test, assert_instr(clz))]
pub unsafe fn _clz_u64(x: u64) -> u64 {
    x.leading_zeros() as u64
}

#[allow(dead_code)]
extern "C" {
    #[link_name = "llvm.bitreverse.i64"]
    fn rbit_u64(i: i64) -> i64;
}

/// Reverse the bit order.
#[inline(always)]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn _rbit_u64(x: u64) -> u64 {
    rbit_u64(x as i64) as u64
}

/// Counts the leading most significant bits set.
///
/// When all bits of the operand are set it returns the size of the operand in
/// bits.
#[inline(always)]
#[cfg_attr(test, assert_instr(cls))]
pub unsafe fn _cls_u32(x: u32) -> u32 {
    u32::leading_zeros((((((x as i32) >> 31) as u32) ^ x) << 1) | 1) as u32
}

/// Counts the leading most significant bits set.
///
/// When all bits of the operand are set it returns the size of the operand in
/// bits.
#[inline(always)]
#[cfg_attr(test, assert_instr(cls))]
pub unsafe fn _cls_u64(x: u64) -> u64 {
    u64::leading_zeros((((((x as i64) >> 63) as u64) ^ x) << 1) | 1) as u64
}

#[cfg(test)]
mod tests {
    use aarch64::v8;

    #[test]
    fn _rev_u64() {
        unsafe {
            assert_eq!(
                v8::_rev_u64(0b0000_0000_1111_1111_0000_0000_1111_1111_u64),
                0b1111_1111_0000_0000_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_u64
            );
        }
    }

    #[test]
    fn _clz_u64() {
        unsafe {
            assert_eq!(v8::_clz_u64(0b0000_1010u64), 60u64);
        }
    }

    #[test]
    fn _rbit_u64() {
        unsafe {
            assert_eq!(
                v8::_rbit_u64(0b0000_0000_1111_1101_0000_0000_1111_1111_u64),
                0b1111_1111_0000_0000_1011_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_u64
            );
        }
    }

    #[test]
    fn _cls_u32() {
        unsafe {
            assert_eq!(
                v8::_cls_u32(0b1111_1111_1111_1111_0000_0000_1111_1111_u32),
                15_u32
            );
        }
    }

    #[test]
    fn _cls_u64() {
        unsafe {
            assert_eq!(
                v8::_cls_u64(
                    0b1111_1111_1111_1111_0000_0000_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_u64
                ),
                15_u64
            );
        }
    }
}
