/*!

# Floating-Point Number To Decimal Conversion

Given a floating-point f as `Decoded64`, we compute its decimal representation
with n significant digits stored in array d[0..n-1], and a base-10 exponent k,
such that:

 v = (0.d₀d₁…dₙ₋₁) × 10ᵏ

where d₀ ≠ 0, ensuring 0.1 ≤ mantissa < 1.

The computed v must be the closest such n-digit decimal to f:

 |f − v| ≤ 10ᵏ⁻ⁿ / 2

If two n-digit decimals are equally close, then some tie-breaking mechanism is
used. This ensures that parsing v back recovers f exactly.

In *short* mode, the smallest n ≥ 1 is taken which still satisfies the above,
thus yielding the minimal decimal which round‑trips correctly. Such formatting
matches ordinary intuition. Note how 0.1f32 prints as "0.1".

In *fixed* mode, n is limited by buffer size. The `resolution` parameter may
further restrict n by requiring v to be an integer multiple of 10^{resolution}.
For example, a resolution of -3 causes rounding to 3 decimal places, i.e., have
multiples of 0.001.

# Implementation overview

Floating‑point to decimal conversion is deceptively simple to implement
incorrectly. Russ Cox [demonstrated](https://research.swtch.com/ftoa) how a
slow but correct algorithm can be concise — yet a fast version risks subtle
rounding errors if based on naïve division and modulo.

There are two widely‑known classes of correct algorithms:

- The **Dragon** family, first described by Guy L. Steele Jr. and Jon L. White,
  relies on arbitrary‑precision integer arithmetic to guarantee correct rounding.
  A later improvement was published posthumously by Robert G. Burger and
  R. Kent Dybvig. David Gay’s `dtoa.c` is a well‑known implementation of this
  strategy.

- The **Grisu** family, introduced by Florian Loitsch, uses a fast, integer‑only
  procedure to produce a decimal representation that is **always shortest**.
  The Grisu3 variant actively detects when the result might not satisfy the
  exact rounding condition (|f − v| ≤ 10ᵏ⁻ⁿ / 2).

Our Grisu implementation uses `Option` `None` to indicate uncertainty. A fall
back to Dragon delivers both performance and guaranteed correctness.

*/

#![doc(hidden)]
#![unstable(
    feature = "flt2dec",
    reason = "internals for use within library exclusively",
    issue = "none"
)]

pub mod decoder;
pub mod estimator;

/// Digit-generation algorithms.
pub mod strategy {
    pub mod dragon;
    pub mod grisu;
}

/// The buffer size needed for digits in shortest mode can be calculated as:
/// ceil(bits_in_mantissa * log10(2)) + 1. Decoded64 is limited to f64 (with
/// 53 mantissa bits), thus the value is set to ceil(15.95) + 1.
pub const SHORT_DIGITS_MAX: usize = 17;

/// When `d` contains decimal digits, increase the last digit and propagate carry.
/// Returns a next digit when it causes the length to change.
#[doc(hidden)]
pub fn round_up(d: &mut [u8]) -> Option<u8> {
    match d.iter().rposition(|&c| c != b'9') {
        Some(i) => {
            // d[i+1..n] is all nines
            d[i] += 1;
            d[i + 1..].fill(b'0');
            None
        }
        None if d.is_empty() => {
            // an empty buffer rounds up (a bit strange but reasonable)
            Some(b'1')
        }
        None => {
            // 999..999 rounds to 1000..000 with an increased exponent
            d[0] = b'1';
            d[1..].fill(b'0');
            Some(b'0')
        }
    }
}
