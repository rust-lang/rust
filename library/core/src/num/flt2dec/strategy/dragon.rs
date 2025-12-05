//! Almost direct (but slightly optimized) Rust translation of Figure 3 of "Printing
//! Floating-Point Numbers Quickly and Accurately"[^1].
//!
//! [^1]: Burger, R. G. and Dybvig, R. K. 1996. Printing floating-point numbers
//!   quickly and accurately. SIGPLAN Not. 31, 5 (May. 1996), 108-116.

use crate::cmp::Ordering;
use crate::mem::MaybeUninit;
use crate::num::bignum::{Big32x40 as Big, Digit32 as Digit};
use crate::num::flt2dec::estimator::estimate_scaling_factor;
use crate::num::flt2dec::{Decoded, MAX_SIG_DIGITS, round_up};

static POW10: [Digit; 10] =
    [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000];
// precalculated arrays of `Digit`s for 5^(2^n).
static POW5TO16: [Digit; 2] = [0x86f26fc1, 0x23];
static POW5TO32: [Digit; 3] = [0x85acef81, 0x2d6d415b, 0x4ee];
static POW5TO64: [Digit; 5] = [0xbf6a1f01, 0x6e38ed64, 0xdaa797ed, 0xe93ff9f4, 0x184f03];
static POW5TO128: [Digit; 10] = [
    0x2e953e01, 0x3df9909, 0xf1538fd, 0x2374e42f, 0xd3cff5ec, 0xc404dc08, 0xbccdb0da, 0xa6337f19,
    0xe91f2603, 0x24e,
];
static POW5TO256: [Digit; 19] = [
    0x982e7c01, 0xbed3875b, 0xd8d99f72, 0x12152f87, 0x6bde50c6, 0xcf4a6e70, 0xd595d80f, 0x26b2716e,
    0xadc666b0, 0x1d153624, 0x3c42d35a, 0x63ff540e, 0xcc5573c0, 0x65f9ef17, 0x55bc28f2, 0x80dcc7f7,
    0xf46eeddc, 0x5fdcefce, 0x553f7,
];

#[doc(hidden)]
pub fn mul_pow10(x: &mut Big, n: usize) -> &mut Big {
    debug_assert!(n < 512);
    // Save ourself the left shift for the smallest cases.
    if n < 8 {
        return x.mul_small(POW10[n & 7]);
    }
    // Multiply by the powers of 5 and shift the 2s in at the end.
    // This keeps the intermediate products smaller and faster.
    if n & 7 != 0 {
        x.mul_small(POW10[n & 7] >> (n & 7));
    }
    if n & 8 != 0 {
        x.mul_small(POW10[8] >> 8);
    }
    if n & 16 != 0 {
        x.mul_digits(&POW5TO16);
    }
    if n & 32 != 0 {
        x.mul_digits(&POW5TO32);
    }
    if n & 64 != 0 {
        x.mul_digits(&POW5TO64);
    }
    if n & 128 != 0 {
        x.mul_digits(&POW5TO128);
    }
    if n & 256 != 0 {
        x.mul_digits(&POW5TO256);
    }
    x.mul_pow2(n)
}

fn div_2pow10(x: &mut Big, mut n: usize) -> &mut Big {
    let largest = POW10.len() - 1;
    while n > largest {
        x.div_rem_small(POW10[largest]);
        n -= largest;
    }
    x.div_rem_small(POW10[n] << 1);
    x
}

// only usable when `x < 16 * scale`; `scaleN` should be `scale.mul_small(N)`
fn div_rem_upto_16<'a>(
    x: &'a mut Big,
    scale: &Big,
    scale2: &Big,
    scale4: &Big,
    scale8: &Big,
) -> (u8, &'a mut Big) {
    let mut d = 0;
    if *x >= *scale8 {
        x.sub(scale8);
        d += 8;
    }
    if *x >= *scale4 {
        x.sub(scale4);
        d += 4;
    }
    if *x >= *scale2 {
        x.sub(scale2);
        d += 2;
    }
    if *x >= *scale {
        x.sub(scale);
        d += 1;
    }
    debug_assert!(*x < *scale);
    (d, x)
}

/// The shortest mode implementation for Dragon.
pub fn format_shortest<'a>(
    d: &Decoded,
    buf: &'a mut [MaybeUninit<u8>],
) -> (/*digits*/ &'a [u8], /*exp*/ i16) {
    // the number `v` to format is known to be:
    // - equal to `mant * 2^exp`;
    // - preceded by `(mant - 2 * minus) * 2^exp` in the original type; and
    // - followed by `(mant + 2 * plus) * 2^exp` in the original type.
    //
    // obviously, `minus` and `plus` cannot be zero. (for infinities, we use out-of-range values.)
    // also we assume that at least one digit is generated, i.e., `mant` cannot be zero too.
    //
    // this also means that any number between `low = (mant - minus) * 2^exp` and
    // `high = (mant + plus) * 2^exp` will map to this exact floating point number,
    // with bounds included when the original mantissa was even (i.e., `!mant_was_odd`).

    assert!(d.mant > 0);
    assert!(d.minus > 0);
    assert!(d.plus > 0);
    assert!(d.mant.checked_add(d.plus).is_some());
    assert!(d.mant.checked_sub(d.minus).is_some());
    assert!(buf.len() >= MAX_SIG_DIGITS);

    // `a.cmp(&b) < rounding` is `if d.inclusive {a <= b} else {a < b}`
    let rounding = if d.inclusive { Ordering::Greater } else { Ordering::Equal };

    // estimate `k_0` from original inputs satisfying `10^(k_0-1) < high <= 10^(k_0+1)`.
    // the tight bound `k` satisfying `10^(k-1) < high <= 10^k` is calculated later.
    let mut k = estimate_scaling_factor(d.mant + d.plus, d.exp);

    // convert `{mant, plus, minus} * 2^exp` into the fractional form so that:
    // - `v = mant / scale`
    // - `low = (mant - minus) / scale`
    // - `high = (mant + plus) / scale`
    let mut mant = Big::from_u64(d.mant);
    let mut minus = Big::from_u64(d.minus);
    let mut plus = Big::from_u64(d.plus);
    let mut scale = Big::from_small(1);
    if d.exp < 0 {
        scale.mul_pow2(-d.exp as usize);
    } else {
        mant.mul_pow2(d.exp as usize);
        minus.mul_pow2(d.exp as usize);
        plus.mul_pow2(d.exp as usize);
    }

    // divide `mant` by `10^k`. now `scale / 10 < mant + plus <= scale * 10`.
    if k >= 0 {
        mul_pow10(&mut scale, k as usize);
    } else {
        mul_pow10(&mut mant, -k as usize);
        mul_pow10(&mut minus, -k as usize);
        mul_pow10(&mut plus, -k as usize);
    }

    // fixup when `mant + plus > scale` (or `>=`).
    // we are not actually modifying `scale`, since we can skip the initial multiplication instead.
    // now `scale < mant + plus <= scale * 10` and we are ready to generate digits.
    //
    // note that `d[0]` *can* be zero, when `scale - plus < mant < scale`.
    // in this case rounding-up condition (`up` below) will be triggered immediately.
    if scale.cmp(mant.clone().add(&plus)) < rounding {
        // equivalent to scaling `scale` by 10
        k += 1;
    } else {
        mant.mul_small(10);
        minus.mul_small(10);
        plus.mul_small(10);
    }

    // cache `(2, 4, 8) * scale` for digit generation.
    let mut scale2 = scale.clone();
    scale2.mul_pow2(1);
    let mut scale4 = scale.clone();
    scale4.mul_pow2(2);
    let mut scale8 = scale.clone();
    scale8.mul_pow2(3);

    let mut down;
    let mut up;
    let mut i = 0;
    loop {
        // invariants, where `d[0..n-1]` are digits generated so far:
        // - `v = mant / scale * 10^(k-n-1) + d[0..n-1] * 10^(k-n)`
        // - `v - low = minus / scale * 10^(k-n-1)`
        // - `high - v = plus / scale * 10^(k-n-1)`
        // - `(mant + plus) / scale <= 10` (thus `mant / scale < 10`)
        // where `d[i..j]` is a shorthand for `d[i] * 10^(j-i) + ... + d[j-1] * 10 + d[j]`.

        // generate one digit: `d[n] = floor(mant / scale) < 10`.
        let (d, _) = div_rem_upto_16(&mut mant, &scale, &scale2, &scale4, &scale8);
        debug_assert!(d < 10);
        buf[i] = MaybeUninit::new(b'0' + d);
        i += 1;

        // this is a simplified description of the modified Dragon algorithm.
        // many intermediate derivations and completeness arguments are omitted for convenience.
        //
        // start with modified invariants, as we've updated `n`:
        // - `v = mant / scale * 10^(k-n) + d[0..n-1] * 10^(k-n)`
        // - `v - low = minus / scale * 10^(k-n)`
        // - `high - v = plus / scale * 10^(k-n)`
        //
        // assume that `d[0..n-1]` is the shortest representation between `low` and `high`,
        // i.e., `d[0..n-1]` satisfies both of the following but `d[0..n-2]` doesn't:
        // - `low < d[0..n-1] * 10^(k-n) < high` (bijectivity: digits round to `v`); and
        // - `abs(v / 10^(k-n) - d[0..n-1]) <= 1/2` (the last digit is correct).
        //
        // the second condition simplifies to `2 * mant <= scale`.
        // solving invariants in terms of `mant`, `low` and `high` yields
        // a simpler version of the first condition: `-plus < mant < minus`.
        // since `-plus < 0 <= mant`, we have the correct shortest representation
        // when `mant < minus` and `2 * mant <= scale`.
        // (the former becomes `mant <= minus` when the original mantissa is even.)
        //
        // when the second doesn't hold (`2 * mant > scale`), we need to increase the last digit.
        // this is enough for restoring that condition: we already know that
        // the digit generation guarantees `0 <= v / 10^(k-n) - d[0..n-1] < 1`.
        // in this case, the first condition becomes `-plus < mant - scale < minus`.
        // since `mant < scale` after the generation, we have `scale < mant + plus`.
        // (again, this becomes `scale <= mant + plus` when the original mantissa is even.)
        //
        // in short:
        // - stop and round `down` (keep digits as is) when `mant < minus` (or `<=`).
        // - stop and round `up` (increase the last digit) when `scale < mant + plus` (or `<=`).
        // - keep generating otherwise.
        down = mant.cmp(&minus) < rounding;
        up = scale.cmp(mant.clone().add(&plus)) < rounding;
        if down || up {
            break;
        } // we have the shortest representation, proceed to the rounding

        // restore the invariants.
        // this makes the algorithm always terminating: `minus` and `plus` always increases,
        // but `mant` is clipped modulo `scale` and `scale` is fixed.
        mant.mul_small(10);
        minus.mul_small(10);
        plus.mul_small(10);
    }

    // rounding up happens when
    // i) only the rounding-up condition was triggered, or
    // ii) both conditions were triggered and tie breaking prefers rounding up.
    if up && (!down || *mant.mul_pow2(1) >= scale) {
        // if rounding up changes the length, the exponent should also change.
        // it seems that this condition is very hard to satisfy (possibly impossible),
        // but we are just being safe and consistent here.
        // SAFETY: we initialized that memory above.
        if let Some(c) = round_up(unsafe { buf[..i].assume_init_mut() }) {
            buf[i] = MaybeUninit::new(c);
            i += 1;
            k += 1;
        }
    }

    // SAFETY: we initialized that memory above.
    (unsafe { buf[..i].assume_init_ref() }, k)
}

/// The exact and fixed mode implementation for Dragon.
pub fn format_exact<'a>(
    d: &Decoded,
    buf: &'a mut [MaybeUninit<u8>],
    limit: i16,
) -> (/*digits*/ &'a [u8], /*exp*/ i16) {
    assert!(d.mant > 0);
    assert!(d.minus > 0);
    assert!(d.plus > 0);
    assert!(d.mant.checked_add(d.plus).is_some());
    assert!(d.mant.checked_sub(d.minus).is_some());

    // estimate `k_0` from original inputs satisfying `10^(k_0-1) < v <= 10^(k_0+1)`.
    let mut k = estimate_scaling_factor(d.mant, d.exp);

    // `v = mant / scale`.
    let mut mant = Big::from_u64(d.mant);
    let mut scale = Big::from_small(1);
    if d.exp < 0 {
        scale.mul_pow2(-d.exp as usize);
    } else {
        mant.mul_pow2(d.exp as usize);
    }

    // divide `mant` by `10^k`. now `scale / 10 < mant <= scale * 10`.
    if k >= 0 {
        mul_pow10(&mut scale, k as usize);
    } else {
        mul_pow10(&mut mant, -k as usize);
    }

    // fixup when `mant + plus >= scale`, where `plus / scale = 10^-buf.len() / 2`.
    // in order to keep the fixed-size bignum, we actually use `mant + floor(plus) >= scale`.
    // we are not actually modifying `scale`, since we can skip the initial multiplication instead.
    // again with the shortest algorithm, `d[0]` can be zero but will be eventually rounded up.
    if *div_2pow10(&mut scale.clone(), buf.len()).add(&mant) >= scale {
        // equivalent to scaling `scale` by 10
        k += 1;
    } else {
        mant.mul_small(10);
    }

    // if we are working with the last-digit limitation, we need to shorten the buffer
    // before the actual rendering in order to avoid double rounding.
    // note that we have to enlarge the buffer again when rounding up happens!
    let mut len = if k < limit {
        // oops, we cannot even produce *one* digit.
        // this is possible when, say, we've got something like 9.5 and it's being rounded to 10.
        // we return an empty buffer, with an exception of the later rounding-up case
        // which occurs when `k == limit` and has to produce exactly one digit.
        0
    } else if ((k as i32 - limit as i32) as usize) < buf.len() {
        (k - limit) as usize
    } else {
        buf.len()
    };

    if len > 0 {
        // cache `(2, 4, 8) * scale` for digit generation.
        // (this can be expensive, so do not calculate them when the buffer is empty.)
        let mut scale2 = scale.clone();
        scale2.mul_pow2(1);
        let mut scale4 = scale.clone();
        scale4.mul_pow2(2);
        let mut scale8 = scale.clone();
        scale8.mul_pow2(3);

        for i in 0..len {
            if mant.is_zero() {
                // following digits are all zeroes, we stop here
                // do *not* try to perform rounding! rather, fill remaining digits.
                for c in &mut buf[i..len] {
                    *c = MaybeUninit::new(b'0');
                }
                // SAFETY: we initialized that memory above.
                return (unsafe { buf[..len].assume_init_ref() }, k);
            }

            let mut d = 0;
            if mant >= scale8 {
                mant.sub(&scale8);
                d += 8;
            }
            if mant >= scale4 {
                mant.sub(&scale4);
                d += 4;
            }
            if mant >= scale2 {
                mant.sub(&scale2);
                d += 2;
            }
            if mant >= scale {
                mant.sub(&scale);
                d += 1;
            }
            debug_assert!(mant < scale);
            debug_assert!(d < 10);
            buf[i] = MaybeUninit::new(b'0' + d);
            mant.mul_small(10);
        }
    }

    // rounding up if we stop in the middle of digits
    // if the following digits are exactly 5000..., check the prior digit and try to
    // round to even (i.e., avoid rounding up when the prior digit is even).
    let order = mant.cmp(scale.mul_small(5));
    if order == Ordering::Greater
        || (order == Ordering::Equal
            // SAFETY: `buf[len-1]` is initialized.
            && len > 0 && unsafe { buf[len - 1].assume_init() } & 1 == 1)
    {
        // if rounding up changes the length, the exponent should also change.
        // but we've been requested a fixed number of digits, so do not alter the buffer...
        // SAFETY: we initialized that memory above.
        if let Some(c) = round_up(unsafe { buf[..len].assume_init_mut() }) {
            // ...unless we've been requested the fixed precision instead.
            // we also need to check that, if the original buffer was empty,
            // the additional digit can only be added when `k == limit` (edge case).
            k += 1;
            if k > limit && len < buf.len() {
                buf[len] = MaybeUninit::new(c);
                len += 1;
            }
        }
    }

    // SAFETY: we initialized that memory above.
    (unsafe { buf[..len].assume_init_ref() }, k)
}
