//! These functions use the [Karatsuba square root algorithm][1] to compute the
//! [integer square root][2] for the primitive integer types.
//!
//! The signed integer functions can only handle **nonnegative** inputs, so
//! that must be checked before calling those.
//!
//! [1]: <https://web.archive.org/web/20230511212802/https://inria.hal.science/inria-00072854v1/file/RR-3805.pdf>
//! "Paul Zimmermann. Karatsuba Square Root. \[Research Report\] RR-3805,
//! INRIA. 1999, pp.8. (inria-00072854)"
//! [2]: <https://en.wikipedia.org/wiki/Integer_square_root>
//! "Wikipedia contributors. Integer square root. Wikipedia, The Free
//! Encyclopedia."

/// This array stores the [integer square roots][1] and remainders of each
/// [`u8`](prim@u8) value. For example, `U8_ISQRT_WITH_REMAINDER[17]` will be
/// `(4, 1)` because the integer square root of 17 is 4 and because 17 is 1
/// higher than 4 squared.
///
/// [1]: <https://en.wikipedia.org/wiki/Integer_square_root>
/// "Wikipedia contributors. Integer square root. Wikipedia, The Free
/// Encyclopedia."
const U8_ISQRT_WITH_REMAINDER: [(u8, u8); 256] = {
    let mut result = [(0, 0); 256];

    let mut n: usize = 0;
    let mut isqrt_n: usize = 0;
    while n < result.len() {
        result[n] = (isqrt_n as u8, (n - isqrt_n.pow(2)) as u8);

        n += 1;
        if n == (isqrt_n + 1).pow(2) {
            isqrt_n += 1;
        }
    }

    result
};

/// Returns the [integer square root][1] of any [`u8`](prim@u8) input.
///
/// [1]: <https://en.wikipedia.org/wiki/Integer_square_root>
/// "Wikipedia contributors. Integer square root. Wikipedia, The Free
/// Encyclopedia."
#[must_use = "this returns the result of the operation, \
              without modifying the original"]
#[inline(always)]
pub const fn u8(n: u8) -> u8 {
    U8_ISQRT_WITH_REMAINDER[n as usize].0
}

/// Returns the [integer square root][1] of any [`usize`](prim@usize) input.
///
/// [1]: <https://en.wikipedia.org/wiki/Integer_square_root>
/// "Wikipedia contributors. Integer square root. Wikipedia, The Free
/// Encyclopedia."
#[must_use = "this returns the result of the operation, \
              without modifying the original"]
#[inline(always)]
pub const fn usize(n: usize) -> usize {
    #[cfg(target_pointer_width = "16")]
    {
        u16(n as u16) as usize
    }

    #[cfg(target_pointer_width = "32")]
    {
        u32(n as u32) as usize
    }

    #[cfg(target_pointer_width = "64")]
    {
        u64(n as u64) as usize
    }
}

/// Generates an `i*` function that returns the [integer square root][1] of any
/// **nonnegative** input of a specific signed integer type.
///
/// [1]: <https://en.wikipedia.org/wiki/Integer_square_root>
/// "Wikipedia contributors. Integer square root. Wikipedia, The Free
/// Encyclopedia."
macro_rules! signed_fn {
    ($SignedT:ident, $UnsignedT:ident) => {
        /// Returns the [integer square root][1] of any **nonnegative**
        #[doc = concat!("[`", stringify!($SignedT), "`](prim@", stringify!($SignedT), ")")]
        /// input.
        ///
        /// # Safety
        ///
        /// This results in undefined behavior when the input is negative.
        ///
        /// [1]: <https://en.wikipedia.org/wiki/Integer_square_root>
        /// "Wikipedia contributors. Integer square root. Wikipedia, The Free
        /// Encyclopedia."
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const unsafe fn $SignedT(n: $SignedT) -> $SignedT {
            $UnsignedT(n as $UnsignedT) as $SignedT
        }
    };
}

signed_fn!(i8, u8);
signed_fn!(i16, u16);
signed_fn!(i32, u32);
signed_fn!(i64, u64);
signed_fn!(i128, u128);

/// Generates a `u*` function that returns the [integer square root][1] of any
/// input of a specific unsigned integer type.
///
/// [1]: <https://en.wikipedia.org/wiki/Integer_square_root>
/// "Wikipedia contributors. Integer square root. Wikipedia, The Free
/// Encyclopedia."
macro_rules! unsigned_fn {
    ($UnsignedT:ident, $HalfBitsT:ident, $stages:ident) => {
        /// Returns the [integer square root][1] of any
        #[doc = concat!("[`", stringify!($UnsignedT), "`](prim@", stringify!($UnsignedT), ")")]
        /// input.
        ///
        /// [1]: <https://en.wikipedia.org/wiki/Integer_square_root>
        /// "Wikipedia contributors. Integer square root. Wikipedia, The Free
        /// Encyclopedia."
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        pub const fn $UnsignedT(mut n: $UnsignedT) -> $UnsignedT {
            if n <= <$HalfBitsT>::MAX as $UnsignedT {
                $HalfBitsT(n as $HalfBitsT) as $UnsignedT
            } else {
                const EVEN_MAKING_BITMASK: u32 = !1;
                let normalization_shift = n.leading_zeros() & EVEN_MAKING_BITMASK;
                n <<= normalization_shift;

                let s = $stages!(n);

                let denormalization_shift = normalization_shift >> 1;
                s >> denormalization_shift
            }
        }
    };
}

/// Generates the first stage of the computation after normalization.
macro_rules! first_stage {
    ($original_bits:literal, $n:ident) => {{
        const N_SHIFT: u32 = $original_bits - 8;
        let n = $n >> N_SHIFT;

        U8_ISQRT_WITH_REMAINDER[n as usize]
    }};
}

/// Generates a middle stage of the computation.
macro_rules! middle_stage {
    ($original_bits:literal, $ty:ty, $n:ident, $s:ident, $r:ident) => {{
        // SAFETY: Inform the optimizer that `$s` is nonzero. This will allow
        // it to avoid generating code to handle division-by-zero panics in the
        // divisions below.
        //
        // If the original `$n` is zero, the top of the `unsigned_fn` macro
        // recurses instead of continuing to this point, so the original `$n`
        // wasn't a 0 if we've reached here.
        //
        // Then the `unsigned_fn` macro normalizes `$n` so that at least one of
        // the two most-significant bits is a 1.
        //
        // Then these stages take as many of the most-significant bits of `$n`
        // that fit in this stage's type. For example, the stage that handles
        // `u32` deals with the 32 most-significant bits of `$n`. This means
        // that each stage has at least one 1 bit in `n`'s two most-significant
        // bits, making `n` nonzero.
        //
        // Then, the stage previous to this produces `$s` as the correct
        // integer square root for the previous type. Since it was taking the
        // integer square root of a nonzero number, `$s` will be nonzero.
        unsafe { crate::hint::assert_unchecked($s != 0) };

        const N_SHIFT: u32 = $original_bits - <$ty>::BITS;
        let n = ($n >> N_SHIFT) as $ty;

        const HALF_BITS: u32 = <$ty>::BITS >> 1;
        const QUARTER_BITS: u32 = <$ty>::BITS >> 2;
        const LOWER_HALF_1_BITS: $ty = (1 << HALF_BITS) - 1;
        const LOWEST_QUARTER_1_BITS: $ty = (1 << QUARTER_BITS) - 1;

        let lo = n & LOWER_HALF_1_BITS;
        let numerator = (($r as $ty) << QUARTER_BITS) | (lo >> QUARTER_BITS);
        let denominator = ($s as $ty) << 1;
        let q = numerator / denominator;
        let u = numerator % denominator;

        let mut s = ($s << QUARTER_BITS) as $ty + q;
        let (mut r, overflow) =
            ((u << QUARTER_BITS) | (lo & LOWEST_QUARTER_1_BITS)).overflowing_sub(q * q);
        if overflow {
            r = r.wrapping_add(2 * s - 1);
            s -= 1;
        }
        (s, r)
    }};
}

/// Generates the last stage of the computation before denormalization.
macro_rules! last_stage {
    ($ty:ty, $n:ident, $s:ident, $r:ident) => {{
        // SAFETY: Inform the optimizer that `$s` is nonzero. This will allow
        // it to avoid generating code to handle division-by-zero panics in the
        // divisions below.
        //
        // See the proof in the `middle_stage` macro above.
        unsafe { core::hint::assert_unchecked($s != 0) };

        const HALF_BITS: u32 = <$ty>::BITS >> 1;
        const QUARTER_BITS: u32 = <$ty>::BITS >> 2;
        const LOWER_HALF_1_BITS: $ty = (1 << HALF_BITS) - 1;

        let lo = $n & LOWER_HALF_1_BITS;
        let numerator = (($r as $ty) << QUARTER_BITS) | (lo >> QUARTER_BITS);
        let denominator = ($s as $ty) << 1;

        let q = numerator / denominator;
        let mut s = ($s << QUARTER_BITS) as $ty + q;
        let (s_squared, overflow) = s.overflowing_mul(s);
        if overflow || s_squared > $n {
            s -= 1;
        }
        s
    }};
}

/// Generates the stages of the computation between normalization and
/// denormalization for [`u16`](prim@u16).
macro_rules! u16_stages {
    ($n:ident) => {{
        let (s, r) = first_stage!(16, $n);
        last_stage!(u16, $n, s, r)
    }};
}

/// Generates the stages of the computation between normalization and
/// denormalization for [`u32`](prim@u32).
macro_rules! u32_stages {
    ($n:ident) => {{
        let (s, r) = first_stage!(32, $n);
        let (s, r) = middle_stage!(32, u16, $n, s, r);
        last_stage!(u32, $n, s, r)
    }};
}

/// Generates the stages of the computation between normalization and
/// denormalization for [`u64`](prim@u64).
macro_rules! u64_stages {
    ($n:ident) => {{
        let (s, r) = first_stage!(64, $n);
        let (s, r) = middle_stage!(64, u16, $n, s, r);
        let (s, r) = middle_stage!(64, u32, $n, s, r);
        last_stage!(u64, $n, s, r)
    }};
}

/// Generates the stages of the computation between normalization and
/// denormalization for [`u128`](prim@u128).
macro_rules! u128_stages {
    ($n:ident) => {{
        let (s, r) = first_stage!(128, $n);
        let (s, r) = middle_stage!(128, u16, $n, s, r);
        let (s, r) = middle_stage!(128, u32, $n, s, r);
        let (s, r) = middle_stage!(128, u64, $n, s, r);
        last_stage!(u128, $n, s, r)
    }};
}

unsigned_fn!(u16, u8, u16_stages);
unsigned_fn!(u32, u16, u32_stages);
unsigned_fn!(u64, u32, u64_stages);
unsigned_fn!(u128, u64, u128_stages);

/// Instantiate this panic logic once, rather than for all the isqrt methods
/// on every single primitive type.
#[cold]
#[track_caller]
pub const fn panic_for_negative_argument() -> ! {
    panic!("argument of integer square root cannot be negative")
}
