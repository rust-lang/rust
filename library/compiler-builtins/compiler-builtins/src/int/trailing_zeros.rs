#[cfg(feature = "unstable-public-internals")]
pub use implementation::trailing_zeros;
#[cfg(not(feature = "unstable-public-internals"))]
pub(crate) use implementation::trailing_zeros;

mod implementation {
    use crate::int::{CastFrom, Int};

    /// Returns number of trailing binary zeros in `x`.
    #[allow(dead_code)]
    pub fn trailing_zeros<I: Int>(x: I) -> usize
    where
        u32: CastFrom<I>,
        u16: CastFrom<I>,
        u8: CastFrom<I>,
    {
        let mut x = x;
        let mut r: u32 = 0;
        let mut t: u32;

        const { assert!(I::BITS <= 64) };
        if I::BITS >= 64 {
            r += ((u32::cast_from_lossy(x) == 0) as u32) << 5; // if (x has no 32 small bits) t = 32 else 0
            x >>= r; // remove 32 zero bits
        }

        if I::BITS >= 32 {
            t = ((u16::cast_from_lossy(x) == 0) as u32) << 4; // if (x has no 16 small bits) t = 16 else 0
            r += t;
            x >>= t; // x = [0 - 0xFFFF] + higher garbage bits
        }

        const { assert!(I::BITS >= 16) };
        t = ((u8::cast_from_lossy(x) == 0) as u32) << 3;
        x >>= t; // x = [0 - 0xFF] + higher garbage bits
        r += t;

        let mut x: u8 = x.cast_lossy();

        t = (((x & 0x0F) == 0) as u32) << 2;
        x >>= t; // x = [0 - 0xF] + higher garbage bits
        r += t;

        t = (((x & 0x3) == 0) as u32) << 1;
        x >>= t; // x = [0 - 0x3] + higher garbage bits
        r += t;

        x &= 3;

        r as usize + ((2 - (x >> 1) as usize) & (((x & 1) == 0) as usize).wrapping_neg())
    }
}

intrinsics! {
    /// Returns the number of trailing binary zeros in `x` (32 bit version).
    pub extern "C" fn __ctzsi2(x: u32) -> usize {
        trailing_zeros(x)
    }

    /// Returns the number of trailing binary zeros in `x` (64 bit version).
    pub extern "C" fn __ctzdi2(x: u64) -> usize {
        trailing_zeros(x)
    }

    /// Returns the number of trailing binary zeros in `x` (128 bit version).
    pub extern "C" fn __ctzti2(x: u128) -> usize {
        let lo = x as u64;
        if lo == 0 {
            64 + __ctzdi2((x >> 64) as u64)
        } else {
            __ctzdi2(lo)
        }
    }
}
