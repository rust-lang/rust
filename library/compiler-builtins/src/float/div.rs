// The functions are complex with many branches, and explicit
// `return`s makes it clear where function exit points are
#![allow(clippy::needless_return)]

use crate::float::Float;
use crate::int::{CastInto, DInt, HInt, Int};

fn div32<F: Float>(a: F, b: F) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
    F::Int: HInt,
    <F as Float>::Int: core::ops::Mul,
{
    const NUMBER_OF_HALF_ITERATIONS: usize = 0;
    const NUMBER_OF_FULL_ITERATIONS: usize = 3;
    const USE_NATIVE_FULL_ITERATIONS: bool = true;

    let one = F::Int::ONE;
    let zero = F::Int::ZERO;
    let hw = F::BITS / 2;
    let lo_mask = u32::MAX >> hw;

    let significand_bits = F::SIGNIFICAND_BITS;
    let max_exponent = F::EXPONENT_MAX;

    let exponent_bias = F::EXPONENT_BIAS;

    let implicit_bit = F::IMPLICIT_BIT;
    let significand_mask = F::SIGNIFICAND_MASK;
    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep = exponent_mask;
    let quiet_bit = implicit_bit >> 1;
    let qnan_rep = exponent_mask | quiet_bit;

    #[inline(always)]
    fn negate_u32(a: u32) -> u32 {
        (<i32>::wrapping_neg(a as i32)) as u32
    }

    let a_rep = a.repr();
    let b_rep = b.repr();

    let a_exponent = (a_rep >> significand_bits) & max_exponent.cast();
    let b_exponent = (b_rep >> significand_bits) & max_exponent.cast();
    let quotient_sign = (a_rep ^ b_rep) & sign_bit;

    let mut a_significand = a_rep & significand_mask;
    let mut b_significand = b_rep & significand_mask;
    let mut scale = 0;

    // Detect if a or b is zero, denormal, infinity, or NaN.
    if a_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
        || b_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
    {
        let a_abs = a_rep & abs_mask;
        let b_abs = b_rep & abs_mask;

        // NaN / anything = qNaN
        if a_abs > inf_rep {
            return F::from_repr(a_rep | quiet_bit);
        }
        // anything / NaN = qNaN
        if b_abs > inf_rep {
            return F::from_repr(b_rep | quiet_bit);
        }

        if a_abs == inf_rep {
            if b_abs == inf_rep {
                // infinity / infinity = NaN
                return F::from_repr(qnan_rep);
            } else {
                // infinity / anything else = +/- infinity
                return F::from_repr(a_abs | quotient_sign);
            }
        }

        // anything else / infinity = +/- 0
        if b_abs == inf_rep {
            return F::from_repr(quotient_sign);
        }

        if a_abs == zero {
            if b_abs == zero {
                // zero / zero = NaN
                return F::from_repr(qnan_rep);
            } else {
                // zero / anything else = +/- zero
                return F::from_repr(quotient_sign);
            }
        }

        // anything else / zero = +/- infinity
        if b_abs == zero {
            return F::from_repr(inf_rep | quotient_sign);
        }

        // one or both of a or b is denormal, the other (if applicable) is a
        // normal number.  Renormalize one or both of a and b, and set scale to
        // include the necessary exponent adjustment.
        if a_abs < implicit_bit {
            let (exponent, significand) = F::normalize(a_significand);
            scale += exponent;
            a_significand = significand;
        }

        if b_abs < implicit_bit {
            let (exponent, significand) = F::normalize(b_significand);
            scale -= exponent;
            b_significand = significand;
        }
    }

    // Set the implicit significand bit.  If we fell through from the
    // denormal path it was already set by normalize( ), but setting it twice
    // won't hurt anything.
    a_significand |= implicit_bit;
    b_significand |= implicit_bit;

    let written_exponent: i32 = CastInto::<u32>::cast(
        a_exponent
            .wrapping_sub(b_exponent)
            .wrapping_add(scale.cast()),
    )
    .wrapping_add(exponent_bias) as i32;
    let b_uq1 = b_significand << (F::BITS - significand_bits - 1);

    // Align the significand of b as a UQ1.(n-1) fixed-point number in the range
    // [1.0, 2.0) and get a UQ0.n approximate reciprocal using a small minimax
    // polynomial approximation: x0 = 3/4 + 1/sqrt(2) - b/2.
    // The max error for this approximation is achieved at endpoints, so
    //   abs(x0(b) - 1/b) <= abs(x0(1) - 1/1) = 3/4 - 1/sqrt(2) = 0.04289...,
    // which is about 4.5 bits.
    // The initial approximation is between x0(1.0) = 0.9571... and x0(2.0) = 0.4571...

    // Then, refine the reciprocal estimate using a quadratically converging
    // Newton-Raphson iteration:
    //     x_{n+1} = x_n * (2 - x_n * b)
    //
    // Let b be the original divisor considered "in infinite precision" and
    // obtained from IEEE754 representation of function argument (with the
    // implicit bit set). Corresponds to rep_t-sized b_UQ1 represented in
    // UQ1.(W-1).
    //
    // Let b_hw be an infinitely precise number obtained from the highest (HW-1)
    // bits of divisor significand (with the implicit bit set). Corresponds to
    // half_rep_t-sized b_UQ1_hw represented in UQ1.(HW-1) that is a **truncated**
    // version of b_UQ1.
    //
    // Let e_n := x_n - 1/b_hw
    //     E_n := x_n - 1/b
    // abs(E_n) <= abs(e_n) + (1/b_hw - 1/b)
    //           = abs(e_n) + (b - b_hw) / (b*b_hw)
    //          <= abs(e_n) + 2 * 2^-HW

    // rep_t-sized iterations may be slower than the corresponding half-width
    // variant depending on the handware and whether single/double/quad precision
    // is selected.
    // NB: Using half-width iterations increases computation errors due to
    // rounding, so error estimations have to be computed taking the selected
    // mode into account!

    #[allow(clippy::absurd_extreme_comparisons)]
    let mut x_uq0 = if NUMBER_OF_HALF_ITERATIONS > 0 {
        // Starting with (n-1) half-width iterations
        let b_uq1_hw: u16 =
            (CastInto::<u32>::cast(b_significand) >> (significand_bits + 1 - hw)) as u16;

        // C is (3/4 + 1/sqrt(2)) - 1 truncated to W0 fractional bits as UQ0.HW
        // with W0 being either 16 or 32 and W0 <= HW.
        // That is, C is the aforementioned 3/4 + 1/sqrt(2) constant (from which
        // b/2 is subtracted to obtain x0) wrapped to [0, 1) range.

        // HW is at least 32. Shifting into the highest bits if needed.
        let c_hw = (0x7504_u32 as u16).wrapping_shl(hw.wrapping_sub(32));

        // b >= 1, thus an upper bound for 3/4 + 1/sqrt(2) - b/2 is about 0.9572,
        // so x0 fits to UQ0.HW without wrapping.
        let x_uq0_hw: u16 = {
            let mut x_uq0_hw: u16 = c_hw.wrapping_sub(b_uq1_hw /* exact b_hw/2 as UQ0.HW */);
            // An e_0 error is comprised of errors due to
            // * x0 being an inherently imprecise first approximation of 1/b_hw
            // * C_hw being some (irrational) number **truncated** to W0 bits
            // Please note that e_0 is calculated against the infinitely precise
            // reciprocal of b_hw (that is, **truncated** version of b).
            //
            // e_0 <= 3/4 - 1/sqrt(2) + 2^-W0

            // By construction, 1 <= b < 2
            // f(x)  = x * (2 - b*x) = 2*x - b*x^2
            // f'(x) = 2 * (1 - b*x)
            //
            // On the [0, 1] interval, f(0)   = 0,
            // then it increses until  f(1/b) = 1 / b, maximum on (0, 1),
            // then it decreses to     f(1)   = 2 - b
            //
            // Let g(x) = x - f(x) = b*x^2 - x.
            // On (0, 1/b), g(x) < 0 <=> f(x) > x
            // On (1/b, 1], g(x) > 0 <=> f(x) < x
            //
            // For half-width iterations, b_hw is used instead of b.
            #[allow(clippy::reversed_empty_ranges)]
            for _ in 0..NUMBER_OF_HALF_ITERATIONS {
                // corr_UQ1_hw can be **larger** than 2 - b_hw*x by at most 1*Ulp
                // of corr_UQ1_hw.
                // "0.0 - (...)" is equivalent to "2.0 - (...)" in UQ1.(HW-1).
                // On the other hand, corr_UQ1_hw should not overflow from 2.0 to 0.0 provided
                // no overflow occurred earlier: ((rep_t)x_UQ0_hw * b_UQ1_hw >> HW) is
                // expected to be strictly positive because b_UQ1_hw has its highest bit set
                // and x_UQ0_hw should be rather large (it converges to 1/2 < 1/b_hw <= 1).
                let corr_uq1_hw: u16 =
                    0.wrapping_sub((x_uq0_hw as u32).wrapping_mul(b_uq1_hw.cast()) >> hw) as u16;

                // Now, we should multiply UQ0.HW and UQ1.(HW-1) numbers, naturally
                // obtaining an UQ1.(HW-1) number and proving its highest bit could be
                // considered to be 0 to be able to represent it in UQ0.HW.
                // From the above analysis of f(x), if corr_UQ1_hw would be represented
                // without any intermediate loss of precision (that is, in twice_rep_t)
                // x_UQ0_hw could be at most [1.]000... if b_hw is exactly 1.0 and strictly
                // less otherwise. On the other hand, to obtain [1.]000..., one have to pass
                // 1/b_hw == 1.0 to f(x), so this cannot occur at all without overflow (due
                // to 1.0 being not representable as UQ0.HW).
                // The fact corr_UQ1_hw was virtually round up (due to result of
                // multiplication being **first** truncated, then negated - to improve
                // error estimations) can increase x_UQ0_hw by up to 2*Ulp of x_UQ0_hw.
                x_uq0_hw = ((x_uq0_hw as u32).wrapping_mul(corr_uq1_hw as u32) >> (hw - 1)) as u16;
                // Now, either no overflow occurred or x_UQ0_hw is 0 or 1 in its half_rep_t
                // representation. In the latter case, x_UQ0_hw will be either 0 or 1 after
                // any number of iterations, so just subtract 2 from the reciprocal
                // approximation after last iteration.

                // In infinite precision, with 0 <= eps1, eps2 <= U = 2^-HW:
                // corr_UQ1_hw = 2 - (1/b_hw + e_n) * b_hw + 2*eps1
                //             = 1 - e_n * b_hw + 2*eps1
                // x_UQ0_hw = (1/b_hw + e_n) * (1 - e_n*b_hw + 2*eps1) - eps2
                //          = 1/b_hw - e_n + 2*eps1/b_hw + e_n - e_n^2*b_hw + 2*e_n*eps1 - eps2
                //          = 1/b_hw + 2*eps1/b_hw - e_n^2*b_hw + 2*e_n*eps1 - eps2
                // e_{n+1} = -e_n^2*b_hw + 2*eps1/b_hw + 2*e_n*eps1 - eps2
                //         = 2*e_n*eps1 - (e_n^2*b_hw + eps2) + 2*eps1/b_hw
                //                        \------ >0 -------/   \-- >0 ---/
                // abs(e_{n+1}) <= 2*abs(e_n)*U + max(2*e_n^2 + U, 2 * U)
            }
            // For initial half-width iterations, U = 2^-HW
            // Let  abs(e_n)     <= u_n * U,
            // then abs(e_{n+1}) <= 2 * u_n * U^2 + max(2 * u_n^2 * U^2 + U, 2 * U)
            // u_{n+1} <= 2 * u_n * U + max(2 * u_n^2 * U + 1, 2)

            // Account for possible overflow (see above). For an overflow to occur for the
            // first time, for "ideal" corr_UQ1_hw (that is, without intermediate
            // truncation), the result of x_UQ0_hw * corr_UQ1_hw should be either maximum
            // value representable in UQ0.HW or less by 1. This means that 1/b_hw have to
            // be not below that value (see g(x) above), so it is safe to decrement just
            // once after the final iteration. On the other hand, an effective value of
            // divisor changes after this point (from b_hw to b), so adjust here.
            x_uq0_hw.wrapping_sub(1_u16)
        };

        // Error estimations for full-precision iterations are calculated just
        // as above, but with U := 2^-W and taking extra decrementing into account.
        // We need at least one such iteration.

        // Simulating operations on a twice_rep_t to perform a single final full-width
        // iteration. Using ad-hoc multiplication implementations to take advantage
        // of particular structure of operands.

        let blo: u32 = (CastInto::<u32>::cast(b_uq1)) & lo_mask;
        // x_UQ0 = x_UQ0_hw * 2^HW - 1
        // x_UQ0 * b_UQ1 = (x_UQ0_hw * 2^HW) * (b_UQ1_hw * 2^HW + blo) - b_UQ1
        //
        //   <--- higher half ---><--- lower half --->
        //   [x_UQ0_hw * b_UQ1_hw]
        // +            [  x_UQ0_hw *  blo  ]
        // -                      [      b_UQ1       ]
        // = [      result       ][.... discarded ...]
        let corr_uq1 = negate_u32(
            (x_uq0_hw as u32) * (b_uq1_hw as u32) + (((x_uq0_hw as u32) * (blo)) >> hw) - 1,
        ); // account for *possible* carry
        let lo_corr = corr_uq1 & lo_mask;
        let hi_corr = corr_uq1 >> hw;
        // x_UQ0 * corr_UQ1 = (x_UQ0_hw * 2^HW) * (hi_corr * 2^HW + lo_corr) - corr_UQ1
        let mut x_uq0: <F as Float>::Int = ((((x_uq0_hw as u32) * hi_corr) << 1)
            .wrapping_add(((x_uq0_hw as u32) * lo_corr) >> (hw - 1))
            .wrapping_sub(2))
        .cast(); // 1 to account for the highest bit of corr_UQ1 can be 1
                 // 1 to account for possible carry
                 // Just like the case of half-width iterations but with possibility
                 // of overflowing by one extra Ulp of x_UQ0.
        x_uq0 -= one;
        // ... and then traditional fixup by 2 should work

        // On error estimation:
        // abs(E_{N-1}) <=   (u_{N-1} + 2 /* due to conversion e_n -> E_n */) * 2^-HW
        //                 + (2^-HW + 2^-W))
        // abs(E_{N-1}) <= (u_{N-1} + 3.01) * 2^-HW

        // Then like for the half-width iterations:
        // With 0 <= eps1, eps2 < 2^-W
        // E_N  = 4 * E_{N-1} * eps1 - (E_{N-1}^2 * b + 4 * eps2) + 4 * eps1 / b
        // abs(E_N) <= 2^-W * [ 4 * abs(E_{N-1}) + max(2 * abs(E_{N-1})^2 * 2^W + 4, 8)) ]
        // abs(E_N) <= 2^-W * [ 4 * (u_{N-1} + 3.01) * 2^-HW + max(4 + 2 * (u_{N-1} + 3.01)^2, 8) ]
        x_uq0
    } else {
        // C is (3/4 + 1/sqrt(2)) - 1 truncated to 32 fractional bits as UQ0.n
        let c: <F as Float>::Int = (0x7504F333 << (F::BITS - 32)).cast();
        let x_uq0: <F as Float>::Int = c.wrapping_sub(b_uq1);
        // E_0 <= 3/4 - 1/sqrt(2) + 2 * 2^-32
        x_uq0
    };

    let mut x_uq0 = if USE_NATIVE_FULL_ITERATIONS {
        for _ in 0..NUMBER_OF_FULL_ITERATIONS {
            let corr_uq1: u32 = 0.wrapping_sub(
                ((CastInto::<u32>::cast(x_uq0) as u64) * (CastInto::<u32>::cast(b_uq1) as u64))
                    >> F::BITS,
            ) as u32;
            x_uq0 = ((((CastInto::<u32>::cast(x_uq0) as u64) * (corr_uq1 as u64)) >> (F::BITS - 1))
                as u32)
                .cast();
        }
        x_uq0
    } else {
        // not using native full iterations
        x_uq0
    };

    // Finally, account for possible overflow, as explained above.
    x_uq0 = x_uq0.wrapping_sub(2.cast());

    // u_n for different precisions (with N-1 half-width iterations):
    // W0 is the precision of C
    //   u_0 = (3/4 - 1/sqrt(2) + 2^-W0) * 2^HW

    // Estimated with bc:
    //   define half1(un) { return 2.0 * (un + un^2) / 2.0^hw + 1.0; }
    //   define half2(un) { return 2.0 * un / 2.0^hw + 2.0; }
    //   define full1(un) { return 4.0 * (un + 3.01) / 2.0^hw + 2.0 * (un + 3.01)^2 + 4.0; }
    //   define full2(un) { return 4.0 * (un + 3.01) / 2.0^hw + 8.0; }

    //             | f32 (0 + 3) | f32 (2 + 1)  | f64 (3 + 1)  | f128 (4 + 1)
    // u_0         | < 184224974 | < 2812.1     | < 184224974  | < 791240234244348797
    // u_1         | < 15804007  | < 242.7      | < 15804007   | < 67877681371350440
    // u_2         | < 116308    | < 2.81       | < 116308     | < 499533100252317
    // u_3         | < 7.31      |              | < 7.31       | < 27054456580
    // u_4         |             |              |              | < 80.4
    // Final (U_N) | same as u_3 | < 72         | < 218        | < 13920

    // Add 2 to U_N due to final decrement.

    let reciprocal_precision: <F as Float>::Int = 10.cast();

    // Suppose 1/b - P * 2^-W < x < 1/b + P * 2^-W
    let x_uq0 = x_uq0 - reciprocal_precision;
    // Now 1/b - (2*P) * 2^-W < x < 1/b
    // FIXME Is x_UQ0 still >= 0.5?

    let mut quotient: <F as Float>::Int = x_uq0.widen_mul(a_significand << 1).hi();
    // Now, a/b - 4*P * 2^-W < q < a/b for q=<quotient_UQ1:dummy> in UQ1.(SB+1+W).

    // quotient_UQ1 is in [0.5, 2.0) as UQ1.(SB+1),
    // adjust it to be in [1.0, 2.0) as UQ1.SB.
    let (mut residual, written_exponent) = if quotient < (implicit_bit << 1) {
        // Highest bit is 0, so just reinterpret quotient_UQ1 as UQ1.SB,
        // effectively doubling its value as well as its error estimation.
        let residual_lo = (a_significand << (significand_bits + 1)).wrapping_sub(
            (CastInto::<u32>::cast(quotient).wrapping_mul(CastInto::<u32>::cast(b_significand)))
                .cast(),
        );
        a_significand <<= 1;
        (residual_lo, written_exponent.wrapping_sub(1))
    } else {
        // Highest bit is 1 (the UQ1.(SB+1) value is in [1, 2)), convert it
        // to UQ1.SB by right shifting by 1. Least significant bit is omitted.
        quotient >>= 1;
        let residual_lo = (a_significand << significand_bits).wrapping_sub(
            (CastInto::<u32>::cast(quotient).wrapping_mul(CastInto::<u32>::cast(b_significand)))
                .cast(),
        );
        (residual_lo, written_exponent)
    };

    //drop mutability
    let quotient = quotient;

    // NB: residualLo is calculated above for the normal result case.
    //     It is re-computed on denormal path that is expected to be not so
    //     performance-sensitive.

    // Now, q cannot be greater than a/b and can differ by at most 8*P * 2^-W + 2^-SB
    // Each NextAfter() increments the floating point value by at least 2^-SB
    // (more, if exponent was incremented).
    // Different cases (<---> is of 2^-SB length, * = a/b that is shown as a midpoint):
    //   q
    //   |   | * |   |   |       |       |
    //       <--->      2^t
    //   |   |   |   |   |   *   |       |
    //               q
    // To require at most one NextAfter(), an error should be less than 1.5 * 2^-SB.
    //   (8*P) * 2^-W + 2^-SB < 1.5 * 2^-SB
    //   (8*P) * 2^-W         < 0.5 * 2^-SB
    //   P < 2^(W-4-SB)
    // Generally, for at most R NextAfter() to be enough,
    //   P < (2*R - 1) * 2^(W-4-SB)
    // For f32 (0+3): 10 < 32 (OK)
    // For f32 (2+1): 32 < 74 < 32 * 3, so two NextAfter() are required
    // For f64: 220 < 256 (OK)
    // For f128: 4096 * 3 < 13922 < 4096 * 5 (three NextAfter() are required)

    // If we have overflowed the exponent, return infinity
    if written_exponent >= max_exponent as i32 {
        return F::from_repr(inf_rep | quotient_sign);
    }

    // Now, quotient <= the correctly-rounded result
    // and may need taking NextAfter() up to 3 times (see error estimates above)
    // r = a - b * q
    let abs_result = if written_exponent > 0 {
        let mut ret = quotient & significand_mask;
        ret |= ((written_exponent as u32) << significand_bits).cast();
        residual <<= 1;
        ret
    } else {
        if (significand_bits as i32 + written_exponent) < 0 {
            return F::from_repr(quotient_sign);
        }
        let ret = quotient.wrapping_shr(negate_u32(CastInto::<u32>::cast(written_exponent)) + 1);
        residual = (CastInto::<u32>::cast(
            a_significand.wrapping_shl(
                significand_bits.wrapping_add(CastInto::<u32>::cast(written_exponent)),
            ),
        )
        .wrapping_sub(
            (CastInto::<u32>::cast(ret).wrapping_mul(CastInto::<u32>::cast(b_significand))) << 1,
        ))
        .cast();
        ret
    };
    // Round
    let abs_result = {
        residual += abs_result & one; // tie to even
                                      // The above line conditionally turns the below LT comparison into LTE

        if residual > b_significand {
            abs_result + one
        } else {
            abs_result
        }
    };
    F::from_repr(abs_result | quotient_sign)
}

fn div64<F: Float>(a: F, b: F) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
    u64: CastInto<F::Int>,
    F::Int: CastInto<u64>,
    i64: CastInto<F::Int>,
    F::Int: CastInto<i64>,
    F::Int: HInt,
{
    const NUMBER_OF_HALF_ITERATIONS: usize = 3;
    const NUMBER_OF_FULL_ITERATIONS: usize = 1;
    const USE_NATIVE_FULL_ITERATIONS: bool = false;

    let one = F::Int::ONE;
    let zero = F::Int::ZERO;
    let hw = F::BITS / 2;
    let lo_mask = u64::MAX >> hw;

    let significand_bits = F::SIGNIFICAND_BITS;
    let max_exponent = F::EXPONENT_MAX;

    let exponent_bias = F::EXPONENT_BIAS;

    let implicit_bit = F::IMPLICIT_BIT;
    let significand_mask = F::SIGNIFICAND_MASK;
    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep = exponent_mask;
    let quiet_bit = implicit_bit >> 1;
    let qnan_rep = exponent_mask | quiet_bit;

    #[inline(always)]
    fn negate_u64(a: u64) -> u64 {
        (<i64>::wrapping_neg(a as i64)) as u64
    }

    let a_rep = a.repr();
    let b_rep = b.repr();

    let a_exponent = (a_rep >> significand_bits) & max_exponent.cast();
    let b_exponent = (b_rep >> significand_bits) & max_exponent.cast();
    let quotient_sign = (a_rep ^ b_rep) & sign_bit;

    let mut a_significand = a_rep & significand_mask;
    let mut b_significand = b_rep & significand_mask;
    let mut scale = 0;

    // Detect if a or b is zero, denormal, infinity, or NaN.
    if a_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
        || b_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
    {
        let a_abs = a_rep & abs_mask;
        let b_abs = b_rep & abs_mask;

        // NaN / anything = qNaN
        if a_abs > inf_rep {
            return F::from_repr(a_rep | quiet_bit);
        }
        // anything / NaN = qNaN
        if b_abs > inf_rep {
            return F::from_repr(b_rep | quiet_bit);
        }

        if a_abs == inf_rep {
            if b_abs == inf_rep {
                // infinity / infinity = NaN
                return F::from_repr(qnan_rep);
            } else {
                // infinity / anything else = +/- infinity
                return F::from_repr(a_abs | quotient_sign);
            }
        }

        // anything else / infinity = +/- 0
        if b_abs == inf_rep {
            return F::from_repr(quotient_sign);
        }

        if a_abs == zero {
            if b_abs == zero {
                // zero / zero = NaN
                return F::from_repr(qnan_rep);
            } else {
                // zero / anything else = +/- zero
                return F::from_repr(quotient_sign);
            }
        }

        // anything else / zero = +/- infinity
        if b_abs == zero {
            return F::from_repr(inf_rep | quotient_sign);
        }

        // one or both of a or b is denormal, the other (if applicable) is a
        // normal number.  Renormalize one or both of a and b, and set scale to
        // include the necessary exponent adjustment.
        if a_abs < implicit_bit {
            let (exponent, significand) = F::normalize(a_significand);
            scale += exponent;
            a_significand = significand;
        }

        if b_abs < implicit_bit {
            let (exponent, significand) = F::normalize(b_significand);
            scale -= exponent;
            b_significand = significand;
        }
    }

    // Set the implicit significand bit.  If we fell through from the
    // denormal path it was already set by normalize( ), but setting it twice
    // won't hurt anything.
    a_significand |= implicit_bit;
    b_significand |= implicit_bit;

    let written_exponent: i64 = CastInto::<u64>::cast(
        a_exponent
            .wrapping_sub(b_exponent)
            .wrapping_add(scale.cast()),
    )
    .wrapping_add(exponent_bias as u64) as i64;
    let b_uq1 = b_significand << (F::BITS - significand_bits - 1);

    // Align the significand of b as a UQ1.(n-1) fixed-point number in the range
    // [1.0, 2.0) and get a UQ0.n approximate reciprocal using a small minimax
    // polynomial approximation: x0 = 3/4 + 1/sqrt(2) - b/2.
    // The max error for this approximation is achieved at endpoints, so
    //   abs(x0(b) - 1/b) <= abs(x0(1) - 1/1) = 3/4 - 1/sqrt(2) = 0.04289...,
    // which is about 4.5 bits.
    // The initial approximation is between x0(1.0) = 0.9571... and x0(2.0) = 0.4571...

    // Then, refine the reciprocal estimate using a quadratically converging
    // Newton-Raphson iteration:
    //     x_{n+1} = x_n * (2 - x_n * b)
    //
    // Let b be the original divisor considered "in infinite precision" and
    // obtained from IEEE754 representation of function argument (with the
    // implicit bit set). Corresponds to rep_t-sized b_UQ1 represented in
    // UQ1.(W-1).
    //
    // Let b_hw be an infinitely precise number obtained from the highest (HW-1)
    // bits of divisor significand (with the implicit bit set). Corresponds to
    // half_rep_t-sized b_UQ1_hw represented in UQ1.(HW-1) that is a **truncated**
    // version of b_UQ1.
    //
    // Let e_n := x_n - 1/b_hw
    //     E_n := x_n - 1/b
    // abs(E_n) <= abs(e_n) + (1/b_hw - 1/b)
    //           = abs(e_n) + (b - b_hw) / (b*b_hw)
    //          <= abs(e_n) + 2 * 2^-HW

    // rep_t-sized iterations may be slower than the corresponding half-width
    // variant depending on the handware and whether single/double/quad precision
    // is selected.
    // NB: Using half-width iterations increases computation errors due to
    // rounding, so error estimations have to be computed taking the selected
    // mode into account!

    let mut x_uq0 = if NUMBER_OF_HALF_ITERATIONS > 0 {
        // Starting with (n-1) half-width iterations
        let b_uq1_hw: u32 =
            (CastInto::<u64>::cast(b_significand) >> (significand_bits + 1 - hw)) as u32;

        // C is (3/4 + 1/sqrt(2)) - 1 truncated to W0 fractional bits as UQ0.HW
        // with W0 being either 16 or 32 and W0 <= HW.
        // That is, C is the aforementioned 3/4 + 1/sqrt(2) constant (from which
        // b/2 is subtracted to obtain x0) wrapped to [0, 1) range.

        // HW is at least 32. Shifting into the highest bits if needed.
        let c_hw = (0x7504F333_u64 as u32).wrapping_shl(hw.wrapping_sub(32));

        // b >= 1, thus an upper bound for 3/4 + 1/sqrt(2) - b/2 is about 0.9572,
        // so x0 fits to UQ0.HW without wrapping.
        let x_uq0_hw: u32 = {
            let mut x_uq0_hw: u32 = c_hw.wrapping_sub(b_uq1_hw /* exact b_hw/2 as UQ0.HW */);
            // dbg!(x_uq0_hw);
            // An e_0 error is comprised of errors due to
            // * x0 being an inherently imprecise first approximation of 1/b_hw
            // * C_hw being some (irrational) number **truncated** to W0 bits
            // Please note that e_0 is calculated against the infinitely precise
            // reciprocal of b_hw (that is, **truncated** version of b).
            //
            // e_0 <= 3/4 - 1/sqrt(2) + 2^-W0

            // By construction, 1 <= b < 2
            // f(x)  = x * (2 - b*x) = 2*x - b*x^2
            // f'(x) = 2 * (1 - b*x)
            //
            // On the [0, 1] interval, f(0)   = 0,
            // then it increses until  f(1/b) = 1 / b, maximum on (0, 1),
            // then it decreses to     f(1)   = 2 - b
            //
            // Let g(x) = x - f(x) = b*x^2 - x.
            // On (0, 1/b), g(x) < 0 <=> f(x) > x
            // On (1/b, 1], g(x) > 0 <=> f(x) < x
            //
            // For half-width iterations, b_hw is used instead of b.
            for _ in 0..NUMBER_OF_HALF_ITERATIONS {
                // corr_UQ1_hw can be **larger** than 2 - b_hw*x by at most 1*Ulp
                // of corr_UQ1_hw.
                // "0.0 - (...)" is equivalent to "2.0 - (...)" in UQ1.(HW-1).
                // On the other hand, corr_UQ1_hw should not overflow from 2.0 to 0.0 provided
                // no overflow occurred earlier: ((rep_t)x_UQ0_hw * b_UQ1_hw >> HW) is
                // expected to be strictly positive because b_UQ1_hw has its highest bit set
                // and x_UQ0_hw should be rather large (it converges to 1/2 < 1/b_hw <= 1).
                let corr_uq1_hw: u32 =
                    0.wrapping_sub(((x_uq0_hw as u64).wrapping_mul(b_uq1_hw as u64)) >> hw) as u32;
                // dbg!(corr_uq1_hw);

                // Now, we should multiply UQ0.HW and UQ1.(HW-1) numbers, naturally
                // obtaining an UQ1.(HW-1) number and proving its highest bit could be
                // considered to be 0 to be able to represent it in UQ0.HW.
                // From the above analysis of f(x), if corr_UQ1_hw would be represented
                // without any intermediate loss of precision (that is, in twice_rep_t)
                // x_UQ0_hw could be at most [1.]000... if b_hw is exactly 1.0 and strictly
                // less otherwise. On the other hand, to obtain [1.]000..., one have to pass
                // 1/b_hw == 1.0 to f(x), so this cannot occur at all without overflow (due
                // to 1.0 being not representable as UQ0.HW).
                // The fact corr_UQ1_hw was virtually round up (due to result of
                // multiplication being **first** truncated, then negated - to improve
                // error estimations) can increase x_UQ0_hw by up to 2*Ulp of x_UQ0_hw.
                x_uq0_hw = ((x_uq0_hw as u64).wrapping_mul(corr_uq1_hw as u64) >> (hw - 1)) as u32;
                // dbg!(x_uq0_hw);
                // Now, either no overflow occurred or x_UQ0_hw is 0 or 1 in its half_rep_t
                // representation. In the latter case, x_UQ0_hw will be either 0 or 1 after
                // any number of iterations, so just subtract 2 from the reciprocal
                // approximation after last iteration.

                // In infinite precision, with 0 <= eps1, eps2 <= U = 2^-HW:
                // corr_UQ1_hw = 2 - (1/b_hw + e_n) * b_hw + 2*eps1
                //             = 1 - e_n * b_hw + 2*eps1
                // x_UQ0_hw = (1/b_hw + e_n) * (1 - e_n*b_hw + 2*eps1) - eps2
                //          = 1/b_hw - e_n + 2*eps1/b_hw + e_n - e_n^2*b_hw + 2*e_n*eps1 - eps2
                //          = 1/b_hw + 2*eps1/b_hw - e_n^2*b_hw + 2*e_n*eps1 - eps2
                // e_{n+1} = -e_n^2*b_hw + 2*eps1/b_hw + 2*e_n*eps1 - eps2
                //         = 2*e_n*eps1 - (e_n^2*b_hw + eps2) + 2*eps1/b_hw
                //                        \------ >0 -------/   \-- >0 ---/
                // abs(e_{n+1}) <= 2*abs(e_n)*U + max(2*e_n^2 + U, 2 * U)
            }
            // For initial half-width iterations, U = 2^-HW
            // Let  abs(e_n)     <= u_n * U,
            // then abs(e_{n+1}) <= 2 * u_n * U^2 + max(2 * u_n^2 * U^2 + U, 2 * U)
            // u_{n+1} <= 2 * u_n * U + max(2 * u_n^2 * U + 1, 2)

            // Account for possible overflow (see above). For an overflow to occur for the
            // first time, for "ideal" corr_UQ1_hw (that is, without intermediate
            // truncation), the result of x_UQ0_hw * corr_UQ1_hw should be either maximum
            // value representable in UQ0.HW or less by 1. This means that 1/b_hw have to
            // be not below that value (see g(x) above), so it is safe to decrement just
            // once after the final iteration. On the other hand, an effective value of
            // divisor changes after this point (from b_hw to b), so adjust here.
            x_uq0_hw.wrapping_sub(1_u32)
        };

        // Error estimations for full-precision iterations are calculated just
        // as above, but with U := 2^-W and taking extra decrementing into account.
        // We need at least one such iteration.

        // Simulating operations on a twice_rep_t to perform a single final full-width
        // iteration. Using ad-hoc multiplication implementations to take advantage
        // of particular structure of operands.
        let blo: u64 = (CastInto::<u64>::cast(b_uq1)) & lo_mask;
        // x_UQ0 = x_UQ0_hw * 2^HW - 1
        // x_UQ0 * b_UQ1 = (x_UQ0_hw * 2^HW) * (b_UQ1_hw * 2^HW + blo) - b_UQ1
        //
        //   <--- higher half ---><--- lower half --->
        //   [x_UQ0_hw * b_UQ1_hw]
        // +            [  x_UQ0_hw *  blo  ]
        // -                      [      b_UQ1       ]
        // = [      result       ][.... discarded ...]
        let corr_uq1 = negate_u64(
            (x_uq0_hw as u64) * (b_uq1_hw as u64) + (((x_uq0_hw as u64) * (blo)) >> hw) - 1,
        ); // account for *possible* carry
        let lo_corr = corr_uq1 & lo_mask;
        let hi_corr = corr_uq1 >> hw;
        // x_UQ0 * corr_UQ1 = (x_UQ0_hw * 2^HW) * (hi_corr * 2^HW + lo_corr) - corr_UQ1
        let mut x_uq0: <F as Float>::Int = ((((x_uq0_hw as u64) * hi_corr) << 1)
            .wrapping_add(((x_uq0_hw as u64) * lo_corr) >> (hw - 1))
            .wrapping_sub(2))
        .cast(); // 1 to account for the highest bit of corr_UQ1 can be 1
                 // 1 to account for possible carry
                 // Just like the case of half-width iterations but with possibility
                 // of overflowing by one extra Ulp of x_UQ0.
        x_uq0 -= one;
        // ... and then traditional fixup by 2 should work

        // On error estimation:
        // abs(E_{N-1}) <=   (u_{N-1} + 2 /* due to conversion e_n -> E_n */) * 2^-HW
        //                 + (2^-HW + 2^-W))
        // abs(E_{N-1}) <= (u_{N-1} + 3.01) * 2^-HW

        // Then like for the half-width iterations:
        // With 0 <= eps1, eps2 < 2^-W
        // E_N  = 4 * E_{N-1} * eps1 - (E_{N-1}^2 * b + 4 * eps2) + 4 * eps1 / b
        // abs(E_N) <= 2^-W * [ 4 * abs(E_{N-1}) + max(2 * abs(E_{N-1})^2 * 2^W + 4, 8)) ]
        // abs(E_N) <= 2^-W * [ 4 * (u_{N-1} + 3.01) * 2^-HW + max(4 + 2 * (u_{N-1} + 3.01)^2, 8) ]
        x_uq0
    } else {
        // C is (3/4 + 1/sqrt(2)) - 1 truncated to 64 fractional bits as UQ0.n
        let c: <F as Float>::Int = (0x7504F333 << (F::BITS - 32)).cast();
        let x_uq0: <F as Float>::Int = c.wrapping_sub(b_uq1);
        // E_0 <= 3/4 - 1/sqrt(2) + 2 * 2^-64
        x_uq0
    };

    let mut x_uq0 = if USE_NATIVE_FULL_ITERATIONS {
        for _ in 0..NUMBER_OF_FULL_ITERATIONS {
            let corr_uq1: u64 = 0.wrapping_sub(
                (CastInto::<u64>::cast(x_uq0) * (CastInto::<u64>::cast(b_uq1))) >> F::BITS,
            );
            x_uq0 = ((((CastInto::<u64>::cast(x_uq0) as u128) * (corr_uq1 as u128))
                >> (F::BITS - 1)) as u64)
                .cast();
        }
        x_uq0
    } else {
        // not using native full iterations
        x_uq0
    };

    // Finally, account for possible overflow, as explained above.
    x_uq0 = x_uq0.wrapping_sub(2.cast());

    // u_n for different precisions (with N-1 half-width iterations):
    // W0 is the precision of C
    //   u_0 = (3/4 - 1/sqrt(2) + 2^-W0) * 2^HW

    // Estimated with bc:
    //   define half1(un) { return 2.0 * (un + un^2) / 2.0^hw + 1.0; }
    //   define half2(un) { return 2.0 * un / 2.0^hw + 2.0; }
    //   define full1(un) { return 4.0 * (un + 3.01) / 2.0^hw + 2.0 * (un + 3.01)^2 + 4.0; }
    //   define full2(un) { return 4.0 * (un + 3.01) / 2.0^hw + 8.0; }

    //             | f32 (0 + 3) | f32 (2 + 1)  | f64 (3 + 1)  | f128 (4 + 1)
    // u_0         | < 184224974 | < 2812.1     | < 184224974  | < 791240234244348797
    // u_1         | < 15804007  | < 242.7      | < 15804007   | < 67877681371350440
    // u_2         | < 116308    | < 2.81       | < 116308     | < 499533100252317
    // u_3         | < 7.31      |              | < 7.31       | < 27054456580
    // u_4         |             |              |              | < 80.4
    // Final (U_N) | same as u_3 | < 72         | < 218        | < 13920

    // Add 2 to U_N due to final decrement.

    let reciprocal_precision: <F as Float>::Int = 220.cast();

    // Suppose 1/b - P * 2^-W < x < 1/b + P * 2^-W
    let x_uq0 = x_uq0 - reciprocal_precision;
    // Now 1/b - (2*P) * 2^-W < x < 1/b
    // FIXME Is x_UQ0 still >= 0.5?

    let mut quotient: <F as Float>::Int = x_uq0.widen_mul(a_significand << 1).hi();
    // Now, a/b - 4*P * 2^-W < q < a/b for q=<quotient_UQ1:dummy> in UQ1.(SB+1+W).

    // quotient_UQ1 is in [0.5, 2.0) as UQ1.(SB+1),
    // adjust it to be in [1.0, 2.0) as UQ1.SB.
    let (mut residual, written_exponent) = if quotient < (implicit_bit << 1) {
        // Highest bit is 0, so just reinterpret quotient_UQ1 as UQ1.SB,
        // effectively doubling its value as well as its error estimation.
        let residual_lo = (a_significand << (significand_bits + 1)).wrapping_sub(
            (CastInto::<u64>::cast(quotient).wrapping_mul(CastInto::<u64>::cast(b_significand)))
                .cast(),
        );
        a_significand <<= 1;
        (residual_lo, written_exponent.wrapping_sub(1))
    } else {
        // Highest bit is 1 (the UQ1.(SB+1) value is in [1, 2)), convert it
        // to UQ1.SB by right shifting by 1. Least significant bit is omitted.
        quotient >>= 1;
        let residual_lo = (a_significand << significand_bits).wrapping_sub(
            (CastInto::<u64>::cast(quotient).wrapping_mul(CastInto::<u64>::cast(b_significand)))
                .cast(),
        );
        (residual_lo, written_exponent)
    };

    //drop mutability
    let quotient = quotient;

    // NB: residualLo is calculated above for the normal result case.
    //     It is re-computed on denormal path that is expected to be not so
    //     performance-sensitive.

    // Now, q cannot be greater than a/b and can differ by at most 8*P * 2^-W + 2^-SB
    // Each NextAfter() increments the floating point value by at least 2^-SB
    // (more, if exponent was incremented).
    // Different cases (<---> is of 2^-SB length, * = a/b that is shown as a midpoint):
    //   q
    //   |   | * |   |   |       |       |
    //       <--->      2^t
    //   |   |   |   |   |   *   |       |
    //               q
    // To require at most one NextAfter(), an error should be less than 1.5 * 2^-SB.
    //   (8*P) * 2^-W + 2^-SB < 1.5 * 2^-SB
    //   (8*P) * 2^-W         < 0.5 * 2^-SB
    //   P < 2^(W-4-SB)
    // Generally, for at most R NextAfter() to be enough,
    //   P < (2*R - 1) * 2^(W-4-SB)
    // For f32 (0+3): 10 < 32 (OK)
    // For f32 (2+1): 32 < 74 < 32 * 3, so two NextAfter() are required
    // For f64: 220 < 256 (OK)
    // For f128: 4096 * 3 < 13922 < 4096 * 5 (three NextAfter() are required)

    // If we have overflowed the exponent, return infinity
    if written_exponent >= max_exponent as i64 {
        return F::from_repr(inf_rep | quotient_sign);
    }

    // Now, quotient <= the correctly-rounded result
    // and may need taking NextAfter() up to 3 times (see error estimates above)
    // r = a - b * q
    let abs_result = if written_exponent > 0 {
        let mut ret = quotient & significand_mask;
        ret |= ((written_exponent as u64) << significand_bits).cast();
        residual <<= 1;
        ret
    } else {
        if (significand_bits as i64 + written_exponent) < 0 {
            return F::from_repr(quotient_sign);
        }
        let ret =
            quotient.wrapping_shr((negate_u64(CastInto::<u64>::cast(written_exponent)) + 1) as u32);
        residual = (CastInto::<u64>::cast(
            a_significand.wrapping_shl(
                significand_bits.wrapping_add(CastInto::<u32>::cast(written_exponent)),
            ),
        )
        .wrapping_sub(
            (CastInto::<u64>::cast(ret).wrapping_mul(CastInto::<u64>::cast(b_significand))) << 1,
        ))
        .cast();
        ret
    };
    // Round
    let abs_result = {
        residual += abs_result & one; // tie to even
                                      // conditionally turns the below LT comparison into LTE
        if residual > b_significand {
            abs_result + one
        } else {
            abs_result
        }
    };
    F::from_repr(abs_result | quotient_sign)
}

intrinsics! {
    #[arm_aeabi_alias = __aeabi_fdiv]
    pub extern "C" fn __divsf3(a: f32, b: f32) -> f32 {
        div32(a, b)
    }

    #[arm_aeabi_alias = __aeabi_ddiv]
    pub extern "C" fn __divdf3(a: f64, b: f64) -> f64 {
        div64(a, b)
    }

    #[cfg(target_arch = "arm")]
    pub extern "C" fn __divsf3vfp(a: f32, b: f32) -> f32 {
        a / b
    }

    #[cfg(target_arch = "arm")]
    pub extern "C" fn __divdf3vfp(a: f64, b: f64) -> f64 {
        a / b
    }
}
