//! The various algorithms from the paper.

use crate::cmp::min;
use crate::cmp::Ordering::{Less, Equal, Greater};
use crate::num::diy_float::Fp;
use crate::num::dec2flt::table;
use crate::num::dec2flt::rawfp::{self, Unpacked, RawFloat, fp_to_float, next_float, prev_float};
use crate::num::dec2flt::num::{self, Big};

/// Number of significand bits in Fp
const P: u32 = 64;

// We simply store the best approximation for *all* exponents, so the variable "h" and the
// associated conditions can be omitted. This trades performance for a couple kilobytes of space.

fn power_of_ten(e: i16) -> Fp {
    assert!(e >= table::MIN_E);
    let i = e - table::MIN_E;
    let sig = table::POWERS.0[i as usize];
    let exp = table::POWERS.1[i as usize];
    Fp { f: sig, e: exp }
}

// In most architectures, floating point operations have an explicit bit size, therefore the
// precision of the computation is determined on a per-operation basis.
#[cfg(any(not(target_arch="x86"), target_feature="sse2"))]
mod fpu_precision {
    pub fn set_precision<T>() { }
}

// On x86, the x87 FPU is used for float operations if the SSE/SSE2 extensions are not available.
// The x87 FPU operates with 80 bits of precision by default, which means that operations will
// round to 80 bits causing double rounding to happen when values are eventually represented as
// 32/64 bit float values. To overcome this, the FPU control word can be set so that the
// computations are performed in the desired precision.
#[cfg(all(target_arch="x86", not(target_feature="sse2")))]
mod fpu_precision {
    use crate::mem::size_of;

    /// A structure used to preserve the original value of the FPU control word, so that it can be
    /// restored when the structure is dropped.
    ///
    /// The x87 FPU is a 16-bits register whose fields are as follows:
    ///
    /// | 12-15 | 10-11 | 8-9 | 6-7 |  5 |  4 |  3 |  2 |  1 |  0 |
    /// |------:|------:|----:|----:|---:|---:|---:|---:|---:|---:|
    /// |       | RC    | PC  |     | PM | UM | OM | ZM | DM | IM |
    ///
    /// The documentation for all of the fields is available in the IA-32 Architectures Software
    /// Developer's Manual (Volume 1).
    ///
    /// The only field which is relevant for the following code is PC, Precision Control. This
    /// field determines the precision of the operations performed by the  FPU. It can be set to:
    ///  - 0b00, single precision i.e., 32-bits
    ///  - 0b10, double precision i.e., 64-bits
    ///  - 0b11, double extended precision i.e., 80-bits (default state)
    /// The 0b01 value is reserved and should not be used.
    pub struct FPUControlWord(u16);

    fn set_cw(cw: u16) {
        unsafe { asm!("fldcw $0" :: "m" (cw) :: "volatile") }
    }

    /// Sets the precision field of the FPU to `T` and returns a `FPUControlWord`.
    pub fn set_precision<T>() -> FPUControlWord {
        let cw = 0u16;

        // Compute the value for the Precision Control field that is appropriate for `T`.
        let cw_precision = match size_of::<T>() {
            4 => 0x0000, // 32 bits
            8 => 0x0200, // 64 bits
            _ => 0x0300, // default, 80 bits
        };

        // Get the original value of the control word to restore it later, when the
        // `FPUControlWord` structure is dropped
        unsafe { asm!("fnstcw $0" : "=*m" (&cw) ::: "volatile") }

        // Set the control word to the desired precision. This is achieved by masking away the old
        // precision (bits 8 and 9, 0x300) and replacing it with the precision flag computed above.
        set_cw((cw & 0xFCFF) | cw_precision);

        FPUControlWord(cw)
    }

    impl Drop for FPUControlWord {
        fn drop(&mut self) {
            set_cw(self.0)
        }
    }
}

/// The fast path of Bellerophon using machine-sized integers and floats.
///
/// This is extracted into a separate function so that it can be attempted before constructing
/// a bignum.
pub fn fast_path<T: RawFloat>(integral: &[u8], fractional: &[u8], e: i64) -> Option<T> {
    let num_digits = integral.len() + fractional.len();
    // log_10(f64::MAX_SIG) ~ 15.95. We compare the exact value to MAX_SIG near the end,
    // this is just a quick, cheap rejection (and also frees the rest of the code from
    // worrying about underflow).
    if num_digits > 16 {
        return None;
    }
    if e.abs() >= T::CEIL_LOG5_OF_MAX_SIG as i64 {
        return None;
    }
    let f = num::from_str_unchecked(integral.iter().chain(fractional.iter()));
    if f > T::MAX_SIG {
        return None;
    }

    // The fast path crucially depends on arithmetic being rounded to the correct number of bits
    // without any intermediate rounding. On x86 (without SSE or SSE2) this requires the precision
    // of the x87 FPU stack to be changed so that it directly rounds to 64/32 bit.
    // The `set_precision` function takes care of setting the precision on architectures which
    // require setting it by changing the global state (like the control word of the x87 FPU).
    let _cw = fpu_precision::set_precision::<T>();

    // The case e < 0 cannot be folded into the other branch. Negative powers result in
    // a repeating fractional part in binary, which are rounded, which causes real
    // (and occasionally quite significant!) errors in the final result.
    if e >= 0 {
        Some(T::from_int(f) * T::short_fast_pow10(e as usize))
    } else {
        Some(T::from_int(f) / T::short_fast_pow10(e.abs() as usize))
    }
}

/// Algorithm Bellerophon is trivial code justified by non-trivial numeric analysis.
///
/// It rounds ``f`` to a float with 64 bit significand and multiplies it by the best approximation
/// of `10^e` (in the same floating point format). This is often enough to get the correct result.
/// However, when the result is close to halfway between two adjacent (ordinary) floats, the
/// compound rounding error from multiplying two approximation means the result may be off by a
/// few bits. When this happens, the iterative Algorithm R fixes things up.
///
/// The hand-wavy "close to halfway" is made precise by the numeric analysis in the paper.
/// In the words of Clinger:
///
/// > Slop, expressed in units of the least significant bit, is an inclusive bound for the error
/// > accumulated during the floating point calculation of the approximation to f * 10^e. (Slop is
/// > not a bound for the true error, but bounds the difference between the approximation z and
/// > the best possible approximation that uses p bits of significand.)
pub fn bellerophon<T: RawFloat>(f: &Big, e: i16) -> T {
    let slop;
    if f <= &Big::from_u64(T::MAX_SIG) {
        // The cases abs(e) < log5(2^N) are in fast_path()
        slop = if e >= 0 { 0 } else { 3 };
    } else {
        slop = if e >= 0 { 1 } else { 4 };
    }
    let z = rawfp::big_to_fp(f).mul(&power_of_ten(e)).normalize();
    let exp_p_n = 1 << (P - T::SIG_BITS as u32);
    let lowbits: i64 = (z.f % exp_p_n) as i64;
    // Is the slop large enough to make a difference when
    // rounding to n bits?
    if (lowbits - exp_p_n as i64 / 2).abs() <= slop {
        algorithm_r(f, e, fp_to_float(z))
    } else {
        fp_to_float(z)
    }
}

/// An iterative algorithm that improves a floating point approximation of `f * 10^e`.
///
/// Each iteration gets one unit in the last place closer, which of course takes terribly long to
/// converge if `z0` is even mildly off. Luckily, when used as fallback for Bellerophon, the
/// starting approximation is off by at most one ULP.
fn algorithm_r<T: RawFloat>(f: &Big, e: i16, z0: T) -> T {
    let mut z = z0;
    loop {
        let raw = z.unpack();
        let (m, k) = (raw.sig, raw.k);
        let mut x = f.clone();
        let mut y = Big::from_u64(m);

        // Find positive integers `x`, `y` such that `x / y` is exactly `(f * 10^e) / (m * 2^k)`.
        // This not only avoids dealing with the signs of `e` and `k`, we also eliminate the
        // power of two common to `10^e` and `2^k` to make the numbers smaller.
        make_ratio(&mut x, &mut y, e, k);

        let m_digits = [(m & 0xFF_FF_FF_FF) as u32, (m >> 32) as u32];
        // This is written a bit awkwardly because our bignums don't support
        // negative numbers, so we use the absolute value + sign information.
        // The multiplication with m_digits can't overflow. If `x` or `y` are large enough that
        // we need to worry about overflow, then they are also large enough that `make_ratio` has
        // reduced the fraction by a factor of 2^64 or more.
        let (d2, d_negative) = if x >= y {
            // Don't need x any more, save a clone().
            x.sub(&y).mul_pow2(1).mul_digits(&m_digits);
            (x, false)
        } else {
            // Still need y - make a copy.
            let mut y = y.clone();
            y.sub(&x).mul_pow2(1).mul_digits(&m_digits);
            (y, true)
        };

        if d2 < y {
            let mut d2_double = d2;
            d2_double.mul_pow2(1);
            if m == T::MIN_SIG && d_negative && d2_double > y {
                z = prev_float(z);
            } else {
                return z;
            }
        } else if d2 == y {
            if m % 2 == 0 {
                if m == T::MIN_SIG && d_negative {
                    z = prev_float(z);
                } else {
                    return z;
                }
            } else if d_negative {
                z = prev_float(z);
            } else {
                z = next_float(z);
            }
        } else if d_negative {
            z = prev_float(z);
        } else {
            z = next_float(z);
        }
    }
}

/// Given `x = f` and `y = m` where `f` represent input decimal digits as usual and `m` is the
/// significand of a floating point approximation, make the ratio `x / y` equal to
/// `(f * 10^e) / (m * 2^k)`, possibly reduced by a power of two both have in common.
fn make_ratio(x: &mut Big, y: &mut Big, e: i16, k: i16) {
    let (e_abs, k_abs) = (e.abs() as usize, k.abs() as usize);
    if e >= 0 {
        if k >= 0 {
            // x = f * 10^e, y = m * 2^k, except that we reduce the fraction by some power of two.
            let common = min(e_abs, k_abs);
            x.mul_pow5(e_abs).mul_pow2(e_abs - common);
            y.mul_pow2(k_abs - common);
        } else {
            // x = f * 10^e * 2^abs(k), y = m
            // This can't overflow because it requires positive `e` and negative `k`, which can
            // only happen for values extremely close to 1, which means that `e` and `k` will be
            // comparatively tiny.
            x.mul_pow5(e_abs).mul_pow2(e_abs + k_abs);
        }
    } else {
        if k >= 0 {
            // x = f, y = m * 10^abs(e) * 2^k
            // This can't overflow either, see above.
            y.mul_pow5(e_abs).mul_pow2(k_abs + e_abs);
        } else {
            // x = f * 2^abs(k), y = m * 10^abs(e), again reducing by a common power of two.
            let common = min(e_abs, k_abs);
            x.mul_pow2(k_abs - common);
            y.mul_pow5(e_abs).mul_pow2(e_abs - common);
        }
    }
}

/// Conceptually, Algorithm M is the simplest way to convert a decimal to a float.
///
/// We form a ratio that is equal to `f * 10^e`, then throwing in powers of two until it gives
/// a valid float significand. The binary exponent `k` is the number of times we multiplied
/// numerator or denominator by two, i.e., at all times `f * 10^e` equals `(u / v) * 2^k`.
/// When we have found out significand, we only need to round by inspecting the remainder of the
/// division, which is done in helper functions further below.
///
/// This algorithm is super slow, even with the optimization described in `quick_start()`.
/// However, it's the simplest of the algorithms to adapt for overflow, underflow, and subnormal
/// results. This implementation takes over when Bellerophon and Algorithm R are overwhelmed.
/// Detecting underflow and overflow is easy: The ratio still isn't an in-range significand,
/// yet the minimum/maximum exponent has been reached. In the case of overflow, we simply return
/// infinity.
///
/// Handling underflow and subnormals is trickier. One big problem is that, with the minimum
/// exponent, the ratio might still be too large for a significand. See underflow() for details.
pub fn algorithm_m<T: RawFloat>(f: &Big, e: i16) -> T {
    let mut u;
    let mut v;
    let e_abs = e.abs() as usize;
    let mut k = 0;
    if e < 0 {
        u = f.clone();
        v = Big::from_small(1);
        v.mul_pow5(e_abs).mul_pow2(e_abs);
    } else {
        // FIXME possible optimization: generalize big_to_fp so that we can do the equivalent of
        // fp_to_float(big_to_fp(u)) here, only without the double rounding.
        u = f.clone();
        u.mul_pow5(e_abs).mul_pow2(e_abs);
        v = Big::from_small(1);
    }
    quick_start::<T>(&mut u, &mut v, &mut k);
    let mut rem = Big::from_small(0);
    let mut x = Big::from_small(0);
    let min_sig = Big::from_u64(T::MIN_SIG);
    let max_sig = Big::from_u64(T::MAX_SIG);
    loop {
        u.div_rem(&v, &mut x, &mut rem);
        if k == T::MIN_EXP_INT {
            // We have to stop at the minimum exponent, if we wait until `k < T::MIN_EXP_INT`,
            // then we'd be off by a factor of two. Unfortunately this means we have to special-
            // case normal numbers with the minimum exponent.
            // FIXME find a more elegant formulation, but run the `tiny-pow10` test to make sure
            // that it's actually correct!
            if x >= min_sig && x <= max_sig {
                break;
            }
            return underflow(x, v, rem);
        }
        if k > T::MAX_EXP_INT {
            return T::INFINITY;
        }
        if x < min_sig {
            u.mul_pow2(1);
            k -= 1;
        } else if x > max_sig {
            v.mul_pow2(1);
            k += 1;
        } else {
            break;
        }
    }
    let q = num::to_u64(&x);
    let z = rawfp::encode_normal(Unpacked::new(q, k));
    round_by_remainder(v, rem, q, z)
}

/// Skips over most Algorithm M iterations by checking the bit length.
fn quick_start<T: RawFloat>(u: &mut Big, v: &mut Big, k: &mut i16) {
    // The bit length is an estimate of the base two logarithm, and log(u / v) = log(u) - log(v).
    // The estimate is off by at most 1, but always an under-estimate, so the error on log(u)
    // and log(v) are of the same sign and cancel out (if both are large). Therefore the error
    // for log(u / v) is at most one as well.
    // The target ratio is one where u/v is in an in-range significand. Thus our termination
    // condition is log2(u / v) being the significand bits, plus/minus one.
    // FIXME Looking at the second bit could improve the estimate and avoid some more divisions.
    let target_ratio = T::SIG_BITS as i16;
    let log2_u = u.bit_length() as i16;
    let log2_v = v.bit_length() as i16;
    let mut u_shift: i16 = 0;
    let mut v_shift: i16 = 0;
    assert!(*k == 0);
    loop {
        if *k == T::MIN_EXP_INT {
            // Underflow or subnormal. Leave it to the main function.
            break;
        }
        if *k == T::MAX_EXP_INT {
            // Overflow. Leave it to the main function.
            break;
        }
        let log2_ratio = (log2_u + u_shift) - (log2_v + v_shift);
        if log2_ratio < target_ratio - 1 {
            u_shift += 1;
            *k -= 1;
        } else if log2_ratio > target_ratio + 1 {
            v_shift += 1;
            *k += 1;
        } else {
            break;
        }
    }
    u.mul_pow2(u_shift as usize);
    v.mul_pow2(v_shift as usize);
}

fn underflow<T: RawFloat>(x: Big, v: Big, rem: Big) -> T {
    if x < Big::from_u64(T::MIN_SIG) {
        let q = num::to_u64(&x);
        let z = rawfp::encode_subnormal(q);
        return round_by_remainder(v, rem, q, z);
    }
    // Ratio isn't an in-range significand with the minimum exponent, so we need to round off
    // excess bits and adjust the exponent accordingly. The real value now looks like this:
    //
    //        x        lsb
    // /--------------\/
    // 1010101010101010.10101010101010 * 2^k
    // \-----/\-------/ \------------/
    //    q     trunc.    (represented by rem)
    //
    // Therefore, when the rounded-off bits are != 0.5 ULP, they decide the rounding
    // on their own. When they are equal and the remainder is non-zero, the value still
    // needs to be rounded up. Only when the rounded off bits are 1/2 and the remainder
    // is zero, we have a half-to-even situation.
    let bits = x.bit_length();
    let lsb = bits - T::SIG_BITS as usize;
    let q = num::get_bits(&x, lsb, bits);
    let k = T::MIN_EXP_INT + lsb as i16;
    let z = rawfp::encode_normal(Unpacked::new(q, k));
    let q_even = q % 2 == 0;
    match num::compare_with_half_ulp(&x, lsb) {
        Greater => next_float(z),
        Less => z,
        Equal if rem.is_zero() && q_even => z,
        Equal => next_float(z),
    }
}

/// Ordinary round-to-even, obfuscated by having to round based on the remainder of a division.
fn round_by_remainder<T: RawFloat>(v: Big, r: Big, q: u64, z: T) -> T {
    let mut v_minus_r = v;
    v_minus_r.sub(&r);
    if r < v_minus_r {
        z
    } else if r > v_minus_r {
        next_float(z)
    } else if q % 2 == 0 {
        z
    } else {
        next_float(z)
    }
}
