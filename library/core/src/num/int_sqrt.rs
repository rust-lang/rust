/// These functions compute the integer square root of their type, assuming
/// that someone has already checked that the value is nonnegative.

const ISQRT_AND_REMAINDER_8_BIT: [(u8, u8); 256] = {
    let mut result = [(0, 0); 256];

    let mut sqrt = 0;
    let mut i = 0;
    'outer: loop {
        let mut remaining = 2 * sqrt + 1;
        while remaining > 0 {
            result[i as usize] = (sqrt, 2 * sqrt + 1 - remaining);
            i += 1;
            if i >= result.len() {
                break 'outer;
            }
            remaining -= 1;
        }
        sqrt += 1;
    }

    result
};

// `#[inline(always)]` because the programmer-accessible functions will use
// this internally and the contents of this should be inlined there.
#[inline(always)]
pub const fn u8(n: u8) -> u8 {
    ISQRT_AND_REMAINDER_8_BIT[n as usize].0
}

#[inline(always)]
const fn intermediate_u8(n: u8) -> (u8, u8) {
    ISQRT_AND_REMAINDER_8_BIT[n as usize]
}

macro_rules! karatsuba_isqrt {
    ($FullBitsT:ty, $fn:ident, $intermediate_fn:ident, $HalfBitsT:ty, $half_fn:ident, $intermediate_half_fn:ident) => {
        // `#[inline(always)]` because the programmer-accessible functions will
        // use this internally and the contents of this should be inlined
        // there.
        #[inline(always)]
        pub const fn $fn(mut n: $FullBitsT) -> $FullBitsT {
            // Performs a Karatsuba square root.
            // https://web.archive.org/web/20230511212802/https://inria.hal.science/inria-00072854v1/file/RR-3805.pdf

            const HALF_BITS: u32 = <$FullBitsT>::BITS >> 1;
            const QUARTER_BITS: u32 = <$FullBitsT>::BITS >> 2;

            let leading_zeros = n.leading_zeros();
            let result = if leading_zeros >= HALF_BITS {
                $half_fn(n as $HalfBitsT) as $FullBitsT
            } else {
                // Either the most-significant bit or its neighbor must be a one, so we shift left to make that happen.
                let precondition_shift = leading_zeros & (HALF_BITS - 2);
                n <<= precondition_shift;

                let hi = (n >> HALF_BITS) as $HalfBitsT;
                let lo = n & (<$HalfBitsT>::MAX as $FullBitsT);

                let (s_prime, r_prime) = $intermediate_half_fn(hi);

                let numerator = ((r_prime as $FullBitsT) << QUARTER_BITS) | (lo >> QUARTER_BITS);
                let denominator = (s_prime as $FullBitsT) << 1;

                let q = numerator / denominator;
                let u = numerator % denominator;

                let mut s = (s_prime << QUARTER_BITS) as $FullBitsT + q;
                if ((u << QUARTER_BITS) | (lo & ((1 << QUARTER_BITS) - 1))) < q * q {
                    s -= 1;
                }
                s >> (precondition_shift >> 1)
            };

            result
        }

        const fn $intermediate_fn(mut n: $FullBitsT) -> ($FullBitsT, $FullBitsT) {
            // Performs a Karatsuba square root.
            // https://web.archive.org/web/20230511212802/https://inria.hal.science/inria-00072854v1/file/RR-3805.pdf

            const HALF_BITS: u32 = <$FullBitsT>::BITS >> 1;
            const QUARTER_BITS: u32 = <$FullBitsT>::BITS >> 2;

            let leading_zeros = n.leading_zeros();
            let result = if leading_zeros >= HALF_BITS {
                let (s, r) = $intermediate_half_fn(n as $HalfBitsT);
                (s as $FullBitsT, r as $FullBitsT)
            } else {
                // Either the most-significant bit or its neighbor must be a one, so we shift left to make that happen.
                let precondition_shift = leading_zeros & (HALF_BITS - 2);
                n <<= precondition_shift;

                let hi = (n >> HALF_BITS) as $HalfBitsT;
                let lo = n & (<$HalfBitsT>::MAX as $FullBitsT);

                let (s_prime, r_prime) = $intermediate_half_fn(hi);

                let numerator = ((r_prime as $FullBitsT) << QUARTER_BITS) | (lo >> QUARTER_BITS);
                let denominator = (s_prime as $FullBitsT) << 1;

                let q = numerator / denominator;
                let u = numerator % denominator;

                let mut s = (s_prime << QUARTER_BITS) as $FullBitsT + q;
                let (mut r, overflow) =
                    ((u << QUARTER_BITS) | (lo & ((1 << QUARTER_BITS) - 1))).overflowing_sub(q * q);
                if overflow {
                    r = r.wrapping_add((s << 1) - 1);
                    s -= 1;
                }
                (s >> (precondition_shift >> 1), r >> (precondition_shift >> 1))
            };

            result
        }
    };
}

karatsuba_isqrt!(u16, u16, intermediate_u16, u8, u8, intermediate_u8);
karatsuba_isqrt!(u32, u32, intermediate_u32, u16, u16, intermediate_u16);
karatsuba_isqrt!(u64, u64, intermediate_u64, u32, u32, intermediate_u32);
karatsuba_isqrt!(u128, u128, _intermediate_u128, u64, u64, intermediate_u64);

#[cfg(target_pointer_width = "16")]
#[inline(always)]
pub const fn usize(n: usize) -> usize {
    u16(n as u16) as usize
}

#[cfg(target_pointer_width = "32")]
#[inline(always)]
pub const fn usize(n: usize) -> usize {
    u32(n as u32) as usize
}

#[cfg(target_pointer_width = "64")]
#[inline(always)]
pub const fn usize(n: usize) -> usize {
    u64(n as u64) as usize
}

// 0 <= val <= i8::MAX
#[inline(always)]
pub const fn i8(n: i8) -> i8 {
    u8(n as u8) as i8
}

// 0 <= val <= i16::MAX
#[inline(always)]
pub const fn i16(n: i16) -> i16 {
    u16(n as u16) as i16
}

// 0 <= val <= i32::MAX
#[inline(always)]
pub const fn i32(n: i32) -> i32 {
    u32(n as u32) as i32
}

// 0 <= val <= i64::MAX
#[inline(always)]
pub const fn i64(n: i64) -> i64 {
    u64(n as u64) as i64
}

// 0 <= val <= i128::MAX
#[inline(always)]
pub const fn i128(n: i128) -> i128 {
    u128(n as u128) as i128
}

/*
This function is not used.

// 0 <= val <= isize::MAX
#[inline(always)]
pub const fn isize(n: isize) -> isize {
    usize(n as usize) as isize
}
*/

/// Instantiate this panic logic once, rather than for all the ilog methods
/// on every single primitive type.
#[cold]
#[track_caller]
pub const fn panic_for_negative_argument() -> ! {
    panic!("argument of integer square root cannot be negative")
}
