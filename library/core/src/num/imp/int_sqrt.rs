//! These functions use the [Karatsuba square root algorithm][1] and an [inverse square root approximation algorithm][2]
//! to compute the [integer square root](https://en.wikipedia.org/wiki/Integer_square_root)
//! for the primitive integer types.
//!
//! The signed integer functions can only handle **nonnegative** inputs, so
//! that must be checked before calling those.
//!
//! [1]: <https://web.archive.org/web/20230511212802/https://inria.hal.science/inria-00072854v1/file/RR-3805.pdf>
//! "Paul Zimmermann. Karatsuba Square Root. \[Research Report\] RR-3805,
//! INRIA. 1999, pp.8. (inria-00072854)"
//!
//! [2]: <https://inria.hal.science/hal-02092970>
//! "Guillaume Melquiond, Raphaël Rieu-Helft. Formal Verification of a State-of-the-Art Integer Square Root.
//! ARITH-26 2019 - 26th IEEE 26th Symposium on Computer Arithmetic, Jun 2019, Kyoto, Japan.
//! pp.183-186, (10.1109/ARITH.2019.00041). (hal-02092970)"

/// A lookup table that stores the [integer square roots](
/// https://en.wikipedia.org/wiki/Integer_square_root) of each [`u8`](prim@u8) value.
static U8_ISQRT: [u8; 256] = {
    let mut result = [0; 256];

    let mut n: usize = 0;
    let mut isqrt_n: usize = 0;
    while n < result.len() {
        result[n] = isqrt_n as u8;

        n += 1;
        if n == (isqrt_n + 1).pow(2) {
            isqrt_n += 1;
        }
    }

    result
};

/// A lookup table for the initial approximation for the inverse square root.
///
/// An alternative approach for calculating the table using floating point
/// arithmetic is:
/// ```compile_fail
/// let k = table_index + 128;
/// let v = 256.0 / ((k as f64 + 0.5) / 512.0).sqrt();
/// v.round() as u32 - 256
/// ```
/// The division by 512.0 scales the number between 0.25 and 1.0.
/// The multiplication by 256.0 extracts eight fractional bits.
static INV_SQRT: [u8; 384] = {
    let mut arr = [0; 384];

    // The number one in fixed point representation.
    // The shift is 10 bits for the input k and 2 * 9 bits for the result.
    let one = 1u32 << 28;

    let mut i = 0;
    while i < arr.len() {
        // Multiplying the value by two turns the addition by 0.5 into
        // an addition by 1, putting the addition inside the integer domain.
        let k = 2 * (i as u32 + 128) + 1;

        // Squaring the inverse and multiplying with the original
        // number should approximately result in 1. The iteration should
        // therefore stop once the value is relatively close.
        // The input got slightly increased, so the inverse square root
        // must become smaller in value. The precision is increased to ensure
        // correct rounding.
        let mut n = 512;
        while (2 * n - 1) * (2 * n - 1) * k >= one {
            n -= 1;
        }

        // This cast removes the leading bit.
        arr[i] = n as u8;
        i += 1;
    }

    arr
};

/// Returns the [integer square root](
/// https://en.wikipedia.org/wiki/Integer_square_root) of any [`u8`](prim@u8)
/// input.
#[must_use = "this returns the result of the operation, \
              without modifying the original"]
#[inline]
pub(in crate::num) const fn u8(n: u8) -> u8 {
    U8_ISQRT[n as usize]
}

/// Generates a `u*` function that returns the [integer square root](
/// https://en.wikipedia.org/wiki/Integer_square_root) of any input of
/// a specific unsigned integer type.
macro_rules! unsigned_fn {
    ($UnsignedT:ident, $HalfBitsT:ident, $inner:ident) => {
        /// Returns the [integer square root](
        /// https://en.wikipedia.org/wiki/Integer_square_root) of any
        #[doc = concat!("[`", stringify!($UnsignedT), "`](prim@", stringify!($UnsignedT), ")")]
        /// input.
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub(in crate::num) const fn $UnsignedT(mut n: $UnsignedT) -> $UnsignedT {
            if n <= <$HalfBitsT>::MAX as $UnsignedT {
                $HalfBitsT(n as $HalfBitsT) as $UnsignedT
            } else {
                // The normalization shift satisfies the square root
                // algorithm precondition "a₃ ≥ b/4" where a₃ is the most
                // significant quarter of `n`'s bits and b is the number of
                // values that can be represented by that quarter of the bits.
                //
                // b/4 would then be all 0s except the second most significant
                // bit (010...0) in binary. Since a₃ must be at least b/4, a₃'s
                // most significant bit or its neighbor must be a 1. Since a₃'s
                // most significant bits are `n`'s most significant bits, the
                // same applies to `n`.
                //
                // The reason to shift by an even number of bits is because an
                // even number of bits produces the square root shifted to the
                // left by half of the normalization shift:
                //
                // sqrt(n << (2 * p))
                // sqrt(2.pow(2 * p) * n)
                // sqrt(2.pow(2 * p)) * sqrt(n)
                // 2.pow(p) * sqrt(n)
                // sqrt(n) << p
                //
                // Shifting by an odd number of bits leaves an ugly sqrt(2)
                // multiplied in:
                //
                // sqrt(n << (2 * p + 1))
                // sqrt(2.pow(2 * p + 1) * n)
                // sqrt(2 * 2.pow(2 * p) * n)
                // sqrt(2) * sqrt(2.pow(2 * p)) * sqrt(n)
                // sqrt(2) * 2.pow(p) * sqrt(n)
                // sqrt(2) * (sqrt(n) << p)
                const EVEN_MAKING_BITMASK: u32 = !1;
                let normalization_shift = n.leading_zeros() & EVEN_MAKING_BITMASK;
                n <<= normalization_shift;

                // SAFETY: The value has just been normalized.
                let s = unsafe { $inner(n) };

                let denormalization_shift = normalization_shift >> 1;
                s >> denormalization_shift
            }
        }
    };
}

/// Takes the normalized [`u16`](prim@u16) input and gets its normalized
/// [integer square root](https://en.wikipedia.org/wiki/Integer_square_root).
///
/// # Safety
///
/// `n` must be normalized, meaning `n >> (u16::BITS - 2) != 0`.
#[inline]
const unsafe fn u16_inner(n: u16) -> u16 {
    // Calculate the integer square root using a variation of the algorithm
    // described in "Formal Verification of a State-of-the-Art Integer Square Root".
    // This version does not do any Newton iteration because the inverse produced
    // by the table lookup is already good enough for the `u16` type.

    const SHIFT: u32 = 16 - 9;
    let index = ((n >> SHIFT) - 128) as usize;
    // SAFETY: `n` is normalized, so `n >> SHIFT` is greater than or equal to 128.
    // This means that the subtraction won't underflow.
    // Furthermore, the maximum value of nine bits is 511, which means that the
    // subtraction results in a maximum value of 383, below the array length.
    let inv = unsafe { *INV_SQRT.as_ptr().add(index) as u16 | 0x100 };

    let mut c = ((n as u32 * inv as u32) >> 16) as u16;
    let s = c * c;
    // It is crucial that the adjustment steps do not branch, otherwise
    // it would result in this algorithm becoming slower than the Karatsuba algorithm.
    c += (s + 2 * c < n) as u16;
    c -= (s > n) as u16;
    c
}

/// Takes the normalized [`u32`](prim@u32) input and gets its normalized
/// [integer square root](https://en.wikipedia.org/wiki/Integer_square_root).
///
/// # Safety
///
/// `n` must be normalized, meaning `n >> (u32::BITS - 2) != 0`.
#[inline]
const unsafe fn u32_inner(n: u32) -> u32 {
    // Calculate the integer square root using a variation of the algorithm
    // described in "Formal Verification of a State-of-the-Art Integer Square Root".
    // This version does only one Newton iteration because of the smaller size of the`u32` type.

    const SHIFT: u32 = 32 - 9;
    let index = ((n >> SHIFT) - 128) as usize;
    // SAFETY: `n` is normalized, so `n >> SHIFT` is greater than or equal to 128.
    // This means that the subtraction won't underflow.
    // Furthermore, the maximum value of nine bits is 511, which means that the
    // subtraction results in a maximum value of 383, below the array length.
    let inv = unsafe { *INV_SQRT.as_ptr().add(index) as u32 | 0x100 };

    // Newton iteration for the next approximation of the inverse square root.
    // Identical to the iteration for the 64-bit square root, but the constants
    // have been adjusted in order to avoid a bit shift on the original input.
    const SUMMAND: i64 = (1 << 48) - 0x18000;
    let t = (SUMMAND - (inv * inv) as i64 * n as i64) >> 15;
    let inv = ((inv as u64) << 16).wrapping_add((t.wrapping_mul(inv as i64) >> 18) as u64);

    let mut c = ((n as u64 * inv) >> 40) as u32;
    let s = c * c;
    c += (s + 2 * c < n) as u32;
    c
}

/// Takes the normalized [`u64`](prim@u64) input and gets its normalized
/// [integer square root](https://en.wikipedia.org/wiki/Integer_square_root).
///
/// # Safety
///
/// `n` must be normalized, meaning `n >> (u64::BITS - 2) != 0`.
#[inline]
const unsafe fn u64_inner(n: u64) -> u64 {
    // SAFETY: The safety conditions are identical.
    unsafe { u64_inner_rem(n).0 }
}

/// Same as [`u64_inner`], but also returns the remainder `r`
/// such that `n = s * s + r` for a given `n` and its square root `s`.
///
/// # Safety
///
/// `n` must be normalized, meaning `n >> (u64::BITS - 2) != 0`.
#[inline]
const unsafe fn u64_inner_rem(n: u64) -> (u64, u64) {
    // Calculate the integer square root using the algorithm
    // described in "Formal Verification of a State-of-the-Art Integer Square Root".

    const SHIFT: u32 = 64 - 9;
    let index = ((n >> SHIFT) - 128) as usize;
    // SAFETY: `n` is normalized, so `n >> SHIFT` is greater than or equal to 128.
    // This means that the subtraction won't underflow.
    // Furthermore, the maximum value of nine bits is 511, which means that the
    // subtraction results in a maximum value of 383, below the array length.
    let inv = unsafe { *INV_SQRT.as_ptr().add(index) as u64 | 0x100 };

    // Newton iteration for the next approximation of the inverse square root.
    const SUMMAND: i64 = (2 << 48) - 0x30000;
    let n1 = n >> 31;
    let t = (SUMMAND - (inv * inv * n1) as i64) >> 16;
    let inv = (inv << 16).wrapping_add((t.wrapping_mul(inv as i64) >> 18) as u64);

    // Newton iteration for the next approximation of the inverse square root.
    // This will also produce an initial approximation of the integer square root.
    const MAGIC: i64 = 1 << 40;
    let t1 = inv.wrapping_mul(n >> 24);
    let t2 = t1 >> 25;
    let t2 = ((n << 14).wrapping_sub(t2.wrapping_mul(t2)) as i64 - MAGIC) >> 24;
    let r = t1.wrapping_add((t2.wrapping_mul(inv as i64) >> 15) as u64);

    let mut root = r >> 32;
    let mut s = root * root;
    if s + 2 * root < n {
        s += 2 * root + 1;
        root += 1;
    }

    (root, n - s)
}

/// Takes the normalized [`u128`](prim@u128) input and gets its normalized
/// [integer square root](https://en.wikipedia.org/wiki/Integer_square_root).
///
/// # Safety
///
/// `n` must be normalized, meaning `n >> (u128::BITS - 2) != 0`.
#[inline]
const unsafe fn u128_inner(n: u128) -> u128 {
    // Calculate the integer square root using the "Karatsuba Square Root" algorithm.

    // SAFETY: normalization is ensured by the caller.
    let (s, r) = unsafe { u64_inner_rem((n >> 64) as u64) };

    let s2 = (2 * s) as u128;
    // SAFETY: The integer square root of a nonzero number is always nonzero.
    unsafe {
        crate::hint::assert_unchecked(s2 > 0);
    }
    let d = (r as u128) << 32 | (n >> 32) & u32::MAX as u128;
    let q = (d / s2) as u64;
    let u = d % s2;

    let s = (s << 32) as u128 + q as u128;
    let qq = (q as u128) * (q as u128);
    let (_r, c) = ((u << 32) + (n & u32::MAX as u128)).overflowing_sub(qq);

    s - c as u128
}

unsigned_fn!(u16, u8, u16_inner);
unsigned_fn!(u32, u16, u32_inner);
unsigned_fn!(u64, u32, u64_inner);
unsigned_fn!(u128, u64, u128_inner);

/// Instantiate this panic logic once, rather than for all the isqrt methods
/// on every single primitive type.
#[cold]
#[track_caller]
pub(in crate::num) const fn panic_for_negative_argument() -> ! {
    panic!("argument of integer square root cannot be negative")
}
