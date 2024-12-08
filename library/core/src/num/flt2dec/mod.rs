/*!

Floating-point number to decimal conversion routines.

# Problem statement

We are given the floating-point number `v = f * 2^e` with an integer `f`,
and its bounds `minus` and `plus` such that any number between `v - minus` and
`v + plus` will be rounded to `v`. For the simplicity we assume that
this range is exclusive. Then we would like to get the unique decimal
representation `V = 0.d[0..n-1] * 10^k` such that:

- `d[0]` is non-zero.

- It's correctly rounded when parsed back: `v - minus < V < v + plus`.
  Furthermore it is shortest such one, i.e., there is no representation
  with less than `n` digits that is correctly rounded.

- It's closest to the original value: `abs(V - v) <= 10^(k-n) / 2`. Note that
  there might be two representations satisfying this uniqueness requirement,
  in which case some tie-breaking mechanism is used.

We will call this mode of operation as to the *shortest* mode. This mode is used
when there is no additional constraint, and can be thought as a "natural" mode
as it matches the ordinary intuition (it at least prints `0.1f32` as "0.1").

We have two more modes of operation closely related to each other. In these modes
we are given either the number of significant digits `n` or the last-digit
limitation `limit` (which determines the actual `n`), and we would like to get
the representation `V = 0.d[0..n-1] * 10^k` such that:

- `d[0]` is non-zero, unless `n` was zero in which case only `k` is returned.

- It's closest to the original value: `abs(V - v) <= 10^(k-n) / 2`. Again,
  there might be some tie-breaking mechanism.

When `limit` is given but not `n`, we set `n` such that `k - n = limit`
so that the last digit `d[n-1]` is scaled by `10^(k-n) = 10^limit`.
If such `n` is negative, we clip it to zero so that we will only get `k`.
We are also limited by the supplied buffer. This limitation is used to print
the number up to given number of fractional digits without knowing
the correct `k` beforehand.

We will call the mode of operation requiring `n` as to the *exact* mode,
and one requiring `limit` as to the *fixed* mode. The exact mode is a subset of
the fixed mode: the sufficiently large last-digit limitation will eventually fill
the supplied buffer and let the algorithm to return.

# Implementation overview

It is easy to get the floating point printing correct but slow (Russ Cox has
[demonstrated](https://research.swtch.com/ftoa) how it's easy), or incorrect but
fast (naïve division and modulo). But it is surprisingly hard to print
floating point numbers correctly *and* efficiently.

There are two classes of algorithms widely known to be correct.

- The "Dragon" family of algorithm is first described by Guy L. Steele Jr. and
  Jon L. White. They rely on the fixed-size big integer for their correctness.
  A slight improvement was found later, which is posthumously described by
  Robert G. Burger and R. Kent Dybvig. David Gay's `dtoa.c` routine is
  a popular implementation of this strategy.

- The "Grisu" family of algorithm is first described by Florian Loitsch.
  They use very cheap integer-only procedure to determine the close-to-correct
  representation which is at least guaranteed to be shortest. The variant,
  Grisu3, actively detects if the resulting representation is incorrect.

We implement both algorithms with necessary tweaks to suit our requirements.
In particular, published literatures are short of the actual implementation
difficulties like how to avoid arithmetic overflows. Each implementation,
available in `strategy::dragon` and `strategy::grisu` respectively,
extensively describes all necessary justifications and many proofs for them.
(It is still difficult to follow though. You have been warned.)

Both implementations expose two public functions:

- `format_shortest(decoded, buf)`, which always needs at least
  `MAX_SIG_DIGITS` digits of buffer. Implements the shortest mode.

- `format_exact(decoded, buf, limit)`, which accepts as small as
  one digit of buffer. Implements exact and fixed modes.

They try to fill the `u8` buffer with digits and returns the number of digits
written and the exponent `k`. They are total for all finite `f32` and `f64`
inputs (Grisu internally falls back to Dragon if necessary).

The rendered digits are formatted into the actual string form with
four functions:

- `to_shortest_str` prints the shortest representation, which can be padded by
  zeroes to make *at least* given number of fractional digits.

- `to_shortest_exp_str` prints the shortest representation, which can be
  padded by zeroes when its exponent is in the specified ranges,
  or can be printed in the exponential form such as `1.23e45`.

- `to_exact_exp_str` prints the exact representation with given number of
  digits in the exponential form.

- `to_exact_fixed_str` prints the fixed representation with *exactly*
  given number of fractional digits.

They all return a slice of preallocated `Part` array, which corresponds to
the individual part of strings: a fixed string, a part of rendered digits,
a number of zeroes or a small (`u16`) number. The caller is expected to
provide a large enough buffer and `Part` array, and to assemble the final
string from resulting `Part`s itself.

All algorithms and formatting functions are accompanied by extensive tests
in `coretests::num::flt2dec` module. It also shows how to use individual
functions.

*/

// while this is extensively documented, this is in principle private which is
// only made public for testing. do not expose us.
#![doc(hidden)]
#![unstable(
    feature = "flt2dec",
    reason = "internal routines only exposed for testing",
    issue = "none"
)]

pub use self::decoder::{DecodableFloat, Decoded, FullDecoded, decode};
use super::fmt::{Formatted, Part};
use crate::mem::MaybeUninit;

pub mod decoder;
pub mod estimator;

/// Digit-generation algorithms.
pub mod strategy {
    pub mod dragon;
    pub mod grisu;
}

/// The minimum size of buffer necessary for the shortest mode.
///
/// It is a bit non-trivial to derive, but this is one plus the maximal number of
/// significant decimal digits from formatting algorithms with the shortest result.
/// The exact formula is `ceil(# bits in mantissa * log_10 2 + 1)`.
pub const MAX_SIG_DIGITS: usize = 17;

/// When `d` contains decimal digits, increase the last digit and propagate carry.
/// Returns a next digit when it causes the length to change.
#[doc(hidden)]
pub fn round_up(d: &mut [u8]) -> Option<u8> {
    match d.iter().rposition(|&c| c != b'9') {
        Some(i) => {
            // d[i+1..n] is all nines
            d[i] += 1;
            for j in i + 1..d.len() {
                d[j] = b'0';
            }
            None
        }
        None if d.len() > 0 => {
            // 999..999 rounds to 1000..000 with an increased exponent
            d[0] = b'1';
            for j in 1..d.len() {
                d[j] = b'0';
            }
            Some(b'0')
        }
        None => {
            // an empty buffer rounds up (a bit strange but reasonable)
            Some(b'1')
        }
    }
}

/// Formats given decimal digits `0.<...buf...> * 10^exp` into the decimal form
/// with at least given number of fractional digits. The result is stored to
/// the supplied parts array and a slice of written parts is returned.
///
/// `frac_digits` can be less than the number of actual fractional digits in `buf`;
/// it will be ignored and full digits will be printed. It is only used to print
/// additional zeroes after rendered digits. Thus `frac_digits` of 0 means that
/// it will only print given digits and nothing else.
fn digits_to_dec_str<'a>(
    buf: &'a [u8],
    exp: i16,
    frac_digits: usize,
    parts: &'a mut [MaybeUninit<Part<'a>>],
) -> &'a [Part<'a>] {
    assert!(!buf.is_empty());
    assert!(buf[0] > b'0');
    assert!(parts.len() >= 4);

    // if there is the restriction on the last digit position, `buf` is assumed to be
    // left-padded with the virtual zeroes. the number of virtual zeroes, `nzeroes`,
    // equals to `max(0, exp + frac_digits - buf.len())`, so that the position of
    // the last digit `exp - buf.len() - nzeroes` is no more than `-frac_digits`:
    //
    //                       |<-virtual->|
    //       |<---- buf ---->|  zeroes   |     exp
    //    0. 1 2 3 4 5 6 7 8 9 _ _ _ _ _ _ x 10
    //    |                  |           |
    // 10^exp    10^(exp-buf.len())   10^(exp-buf.len()-nzeroes)
    //
    // `nzeroes` is individually calculated for each case in order to avoid overflow.

    if exp <= 0 {
        // the decimal point is before rendered digits: [0.][000...000][1234][____]
        let minus_exp = -(exp as i32) as usize;
        parts[0] = MaybeUninit::new(Part::Copy(b"0."));
        parts[1] = MaybeUninit::new(Part::Zero(minus_exp));
        parts[2] = MaybeUninit::new(Part::Copy(buf));
        if frac_digits > buf.len() && frac_digits - buf.len() > minus_exp {
            parts[3] = MaybeUninit::new(Part::Zero((frac_digits - buf.len()) - minus_exp));
            // SAFETY: we just initialized the elements `..4`.
            unsafe { MaybeUninit::slice_assume_init_ref(&parts[..4]) }
        } else {
            // SAFETY: we just initialized the elements `..3`.
            unsafe { MaybeUninit::slice_assume_init_ref(&parts[..3]) }
        }
    } else {
        let exp = exp as usize;
        if exp < buf.len() {
            // the decimal point is inside rendered digits: [12][.][34][____]
            parts[0] = MaybeUninit::new(Part::Copy(&buf[..exp]));
            parts[1] = MaybeUninit::new(Part::Copy(b"."));
            parts[2] = MaybeUninit::new(Part::Copy(&buf[exp..]));
            if frac_digits > buf.len() - exp {
                parts[3] = MaybeUninit::new(Part::Zero(frac_digits - (buf.len() - exp)));
                // SAFETY: we just initialized the elements `..4`.
                unsafe { MaybeUninit::slice_assume_init_ref(&parts[..4]) }
            } else {
                // SAFETY: we just initialized the elements `..3`.
                unsafe { MaybeUninit::slice_assume_init_ref(&parts[..3]) }
            }
        } else {
            // the decimal point is after rendered digits: [1234][____0000] or [1234][__][.][__].
            parts[0] = MaybeUninit::new(Part::Copy(buf));
            parts[1] = MaybeUninit::new(Part::Zero(exp - buf.len()));
            if frac_digits > 0 {
                parts[2] = MaybeUninit::new(Part::Copy(b"."));
                parts[3] = MaybeUninit::new(Part::Zero(frac_digits));
                // SAFETY: we just initialized the elements `..4`.
                unsafe { MaybeUninit::slice_assume_init_ref(&parts[..4]) }
            } else {
                // SAFETY: we just initialized the elements `..2`.
                unsafe { MaybeUninit::slice_assume_init_ref(&parts[..2]) }
            }
        }
    }
}

/// Formats the given decimal digits `0.<...buf...> * 10^exp` into the exponential
/// form with at least the given number of significant digits. When `upper` is `true`,
/// the exponent will be prefixed by `E`; otherwise that's `e`. The result is
/// stored to the supplied parts array and a slice of written parts is returned.
///
/// `min_digits` can be less than the number of actual significant digits in `buf`;
/// it will be ignored and full digits will be printed. It is only used to print
/// additional zeroes after rendered digits. Thus, `min_digits == 0` means that
/// it will only print the given digits and nothing else.
fn digits_to_exp_str<'a>(
    buf: &'a [u8],
    exp: i16,
    min_ndigits: usize,
    upper: bool,
    parts: &'a mut [MaybeUninit<Part<'a>>],
) -> &'a [Part<'a>] {
    assert!(!buf.is_empty());
    assert!(buf[0] > b'0');
    assert!(parts.len() >= 6);

    let mut n = 0;

    parts[n] = MaybeUninit::new(Part::Copy(&buf[..1]));
    n += 1;

    if buf.len() > 1 || min_ndigits > 1 {
        parts[n] = MaybeUninit::new(Part::Copy(b"."));
        parts[n + 1] = MaybeUninit::new(Part::Copy(&buf[1..]));
        n += 2;
        if min_ndigits > buf.len() {
            parts[n] = MaybeUninit::new(Part::Zero(min_ndigits - buf.len()));
            n += 1;
        }
    }

    // 0.1234 x 10^exp = 1.234 x 10^(exp-1)
    let exp = exp as i32 - 1; // avoid underflow when exp is i16::MIN
    if exp < 0 {
        parts[n] = MaybeUninit::new(Part::Copy(if upper { b"E-" } else { b"e-" }));
        parts[n + 1] = MaybeUninit::new(Part::Num(-exp as u16));
    } else {
        parts[n] = MaybeUninit::new(Part::Copy(if upper { b"E" } else { b"e" }));
        parts[n + 1] = MaybeUninit::new(Part::Num(exp as u16));
    }
    // SAFETY: we just initialized the elements `..n + 2`.
    unsafe { MaybeUninit::slice_assume_init_ref(&parts[..n + 2]) }
}

/// Sign formatting options.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Sign {
    /// Prints `-` for any negative value.
    Minus, // -inf -1 -0  0  1  inf nan
    /// Prints `-` for any negative value, or `+` otherwise.
    MinusPlus, // -inf -1 -0 +0 +1 +inf nan
}

/// Returns the static byte string corresponding to the sign to be formatted.
/// It can be either `""`, `"+"` or `"-"`.
fn determine_sign(sign: Sign, decoded: &FullDecoded, negative: bool) -> &'static str {
    match (*decoded, sign) {
        (FullDecoded::Nan, _) => "",
        (_, Sign::Minus) => {
            if negative {
                "-"
            } else {
                ""
            }
        }
        (_, Sign::MinusPlus) => {
            if negative {
                "-"
            } else {
                "+"
            }
        }
    }
}

/// Formats the given floating point number into the decimal form with at least
/// given number of fractional digits. The result is stored to the supplied parts
/// array while utilizing given byte buffer as a scratch. `upper` is currently
/// unused but left for the future decision to change the case of non-finite values,
/// i.e., `inf` and `nan`. The first part to be rendered is always a `Part::Sign`
/// (which can be an empty string if no sign is rendered).
///
/// `format_shortest` should be the underlying digit-generation function.
/// It should return the part of the buffer that it initialized.
/// You probably would want `strategy::grisu::format_shortest` for this.
///
/// `frac_digits` can be less than the number of actual fractional digits in `v`;
/// it will be ignored and full digits will be printed. It is only used to print
/// additional zeroes after rendered digits. Thus `frac_digits` of 0 means that
/// it will only print given digits and nothing else.
///
/// The byte buffer should be at least `MAX_SIG_DIGITS` bytes long.
/// There should be at least 4 parts available, due to the worst case like
/// `[+][0.][0000][2][0000]` with `frac_digits = 10`.
pub fn to_shortest_str<'a, T, F>(
    mut format_shortest: F,
    v: T,
    sign: Sign,
    frac_digits: usize,
    buf: &'a mut [MaybeUninit<u8>],
    parts: &'a mut [MaybeUninit<Part<'a>>],
) -> Formatted<'a>
where
    T: DecodableFloat,
    F: FnMut(&Decoded, &'a mut [MaybeUninit<u8>]) -> (&'a [u8], i16),
{
    assert!(parts.len() >= 4);
    assert!(buf.len() >= MAX_SIG_DIGITS);

    let (negative, full_decoded) = decode(v);
    let sign = determine_sign(sign, &full_decoded, negative);
    match full_decoded {
        FullDecoded::Nan => {
            parts[0] = MaybeUninit::new(Part::Copy(b"NaN"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Infinite => {
            parts[0] = MaybeUninit::new(Part::Copy(b"inf"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Zero => {
            if frac_digits > 0 {
                // [0.][0000]
                parts[0] = MaybeUninit::new(Part::Copy(b"0."));
                parts[1] = MaybeUninit::new(Part::Zero(frac_digits));
                Formatted {
                    sign,
                    // SAFETY: we just initialized the elements `..2`.
                    parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..2]) },
                }
            } else {
                parts[0] = MaybeUninit::new(Part::Copy(b"0"));
                Formatted {
                    sign,
                    // SAFETY: we just initialized the elements `..1`.
                    parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) },
                }
            }
        }
        FullDecoded::Finite(ref decoded) => {
            let (buf, exp) = format_shortest(decoded, buf);
            Formatted { sign, parts: digits_to_dec_str(buf, exp, frac_digits, parts) }
        }
    }
}

/// Formats the given floating point number into the decimal form or
/// the exponential form, depending on the resulting exponent. The result is
/// stored to the supplied parts array while utilizing given byte buffer
/// as a scratch. `upper` is used to determine the case of non-finite values
/// (`inf` and `nan`) or the case of the exponent prefix (`e` or `E`).
/// The first part to be rendered is always a `Part::Sign` (which can be
/// an empty string if no sign is rendered).
///
/// `format_shortest` should be the underlying digit-generation function.
/// It should return the part of the buffer that it initialized.
/// You probably would want `strategy::grisu::format_shortest` for this.
///
/// The `dec_bounds` is a tuple `(lo, hi)` such that the number is formatted
/// as decimal only when `10^lo <= V < 10^hi`. Note that this is the *apparent* `V`
/// instead of the actual `v`! Thus any printed exponent in the exponential form
/// cannot be in this range, avoiding any confusion.
///
/// The byte buffer should be at least `MAX_SIG_DIGITS` bytes long.
/// There should be at least 6 parts available, due to the worst case like
/// `[+][1][.][2345][e][-][6]`.
pub fn to_shortest_exp_str<'a, T, F>(
    mut format_shortest: F,
    v: T,
    sign: Sign,
    dec_bounds: (i16, i16),
    upper: bool,
    buf: &'a mut [MaybeUninit<u8>],
    parts: &'a mut [MaybeUninit<Part<'a>>],
) -> Formatted<'a>
where
    T: DecodableFloat,
    F: FnMut(&Decoded, &'a mut [MaybeUninit<u8>]) -> (&'a [u8], i16),
{
    assert!(parts.len() >= 6);
    assert!(buf.len() >= MAX_SIG_DIGITS);
    assert!(dec_bounds.0 <= dec_bounds.1);

    let (negative, full_decoded) = decode(v);
    let sign = determine_sign(sign, &full_decoded, negative);
    match full_decoded {
        FullDecoded::Nan => {
            parts[0] = MaybeUninit::new(Part::Copy(b"NaN"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Infinite => {
            parts[0] = MaybeUninit::new(Part::Copy(b"inf"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Zero => {
            parts[0] = if dec_bounds.0 <= 0 && 0 < dec_bounds.1 {
                MaybeUninit::new(Part::Copy(b"0"))
            } else {
                MaybeUninit::new(Part::Copy(if upper { b"0E0" } else { b"0e0" }))
            };
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Finite(ref decoded) => {
            let (buf, exp) = format_shortest(decoded, buf);
            let vis_exp = exp as i32 - 1;
            let parts = if dec_bounds.0 as i32 <= vis_exp && vis_exp < dec_bounds.1 as i32 {
                digits_to_dec_str(buf, exp, 0, parts)
            } else {
                digits_to_exp_str(buf, exp, 0, upper, parts)
            };
            Formatted { sign, parts }
        }
    }
}

/// Returns a rather crude approximation (upper bound) for the maximum buffer size
/// calculated from the given decoded exponent.
///
/// The exact limit is:
///
/// - when `exp < 0`, the maximum length is `ceil(log_10 (5^-exp * (2^64 - 1)))`.
/// - when `exp >= 0`, the maximum length is `ceil(log_10 (2^exp * (2^64 - 1)))`.
///
/// `ceil(log_10 (x^exp * (2^64 - 1)))` is less than `ceil(log_10 (2^64 - 1)) +
/// ceil(exp * log_10 x)`, which is in turn less than `20 + (1 + exp * log_10 x)`.
/// We use the facts that `log_10 2 < 5/16` and `log_10 5 < 12/16`, which is
/// enough for our purposes.
///
/// Why do we need this? `format_exact` functions will fill the entire buffer
/// unless limited by the last digit restriction, but it is possible that
/// the number of digits requested is ridiculously large (say, 30,000 digits).
/// The vast majority of buffer will be filled with zeroes, so we don't want to
/// allocate all the buffer beforehand. Consequently, for any given arguments,
/// 826 bytes of buffer should be sufficient for `f64`. Compare this with
/// the actual number for the worst case: 770 bytes (when `exp = -1074`).
fn estimate_max_buf_len(exp: i16) -> usize {
    21 + ((if exp < 0 { -12 } else { 5 } * exp as i32) as usize >> 4)
}

/// Formats given floating point number into the exponential form with
/// exactly given number of significant digits. The result is stored to
/// the supplied parts array while utilizing given byte buffer as a scratch.
/// `upper` is used to determine the case of the exponent prefix (`e` or `E`).
/// The first part to be rendered is always a `Part::Sign` (which can be
/// an empty string if no sign is rendered).
///
/// `format_exact` should be the underlying digit-generation function.
/// It should return the part of the buffer that it initialized.
/// You probably would want `strategy::grisu::format_exact` for this.
///
/// The byte buffer should be at least `ndigits` bytes long unless `ndigits` is
/// so large that only the fixed number of digits will be ever written.
/// (The tipping point for `f64` is about 800, so 1000 bytes should be enough.)
/// There should be at least 6 parts available, due to the worst case like
/// `[+][1][.][2345][e][-][6]`.
pub fn to_exact_exp_str<'a, T, F>(
    mut format_exact: F,
    v: T,
    sign: Sign,
    ndigits: usize,
    upper: bool,
    buf: &'a mut [MaybeUninit<u8>],
    parts: &'a mut [MaybeUninit<Part<'a>>],
) -> Formatted<'a>
where
    T: DecodableFloat,
    F: FnMut(&Decoded, &'a mut [MaybeUninit<u8>], i16) -> (&'a [u8], i16),
{
    assert!(parts.len() >= 6);
    assert!(ndigits > 0);

    let (negative, full_decoded) = decode(v);
    let sign = determine_sign(sign, &full_decoded, negative);
    match full_decoded {
        FullDecoded::Nan => {
            parts[0] = MaybeUninit::new(Part::Copy(b"NaN"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Infinite => {
            parts[0] = MaybeUninit::new(Part::Copy(b"inf"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Zero => {
            if ndigits > 1 {
                // [0.][0000][e0]
                parts[0] = MaybeUninit::new(Part::Copy(b"0."));
                parts[1] = MaybeUninit::new(Part::Zero(ndigits - 1));
                parts[2] = MaybeUninit::new(Part::Copy(if upper { b"E0" } else { b"e0" }));
                Formatted {
                    sign,
                    // SAFETY: we just initialized the elements `..3`.
                    parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..3]) },
                }
            } else {
                parts[0] = MaybeUninit::new(Part::Copy(if upper { b"0E0" } else { b"0e0" }));
                Formatted {
                    sign,
                    // SAFETY: we just initialized the elements `..1`.
                    parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) },
                }
            }
        }
        FullDecoded::Finite(ref decoded) => {
            let maxlen = estimate_max_buf_len(decoded.exp);
            assert!(buf.len() >= ndigits || buf.len() >= maxlen);

            let trunc = if ndigits < maxlen { ndigits } else { maxlen };
            let (buf, exp) = format_exact(decoded, &mut buf[..trunc], i16::MIN);
            Formatted { sign, parts: digits_to_exp_str(buf, exp, ndigits, upper, parts) }
        }
    }
}

/// Formats given floating point number into the decimal form with exactly
/// given number of fractional digits. The result is stored to the supplied parts
/// array while utilizing given byte buffer as a scratch. `upper` is currently
/// unused but left for the future decision to change the case of non-finite values,
/// i.e., `inf` and `nan`. The first part to be rendered is always a `Part::Sign`
/// (which can be an empty string if no sign is rendered).
///
/// `format_exact` should be the underlying digit-generation function.
/// It should return the part of the buffer that it initialized.
/// You probably would want `strategy::grisu::format_exact` for this.
///
/// The byte buffer should be enough for the output unless `frac_digits` is
/// so large that only the fixed number of digits will be ever written.
/// (The tipping point for `f64` is about 800, and 1000 bytes should be enough.)
/// There should be at least 4 parts available, due to the worst case like
/// `[+][0.][0000][2][0000]` with `frac_digits = 10`.
pub fn to_exact_fixed_str<'a, T, F>(
    mut format_exact: F,
    v: T,
    sign: Sign,
    frac_digits: usize,
    buf: &'a mut [MaybeUninit<u8>],
    parts: &'a mut [MaybeUninit<Part<'a>>],
) -> Formatted<'a>
where
    T: DecodableFloat,
    F: FnMut(&Decoded, &'a mut [MaybeUninit<u8>], i16) -> (&'a [u8], i16),
{
    assert!(parts.len() >= 4);

    let (negative, full_decoded) = decode(v);
    let sign = determine_sign(sign, &full_decoded, negative);
    match full_decoded {
        FullDecoded::Nan => {
            parts[0] = MaybeUninit::new(Part::Copy(b"NaN"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Infinite => {
            parts[0] = MaybeUninit::new(Part::Copy(b"inf"));
            // SAFETY: we just initialized the elements `..1`.
            Formatted { sign, parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) } }
        }
        FullDecoded::Zero => {
            if frac_digits > 0 {
                // [0.][0000]
                parts[0] = MaybeUninit::new(Part::Copy(b"0."));
                parts[1] = MaybeUninit::new(Part::Zero(frac_digits));
                Formatted {
                    sign,
                    // SAFETY: we just initialized the elements `..2`.
                    parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..2]) },
                }
            } else {
                parts[0] = MaybeUninit::new(Part::Copy(b"0"));
                Formatted {
                    sign,
                    // SAFETY: we just initialized the elements `..1`.
                    parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) },
                }
            }
        }
        FullDecoded::Finite(ref decoded) => {
            let maxlen = estimate_max_buf_len(decoded.exp);
            assert!(buf.len() >= maxlen);

            // it *is* possible that `frac_digits` is ridiculously large.
            // `format_exact` will end rendering digits much earlier in this case,
            // because we are strictly limited by `maxlen`.
            let limit = if frac_digits < 0x8000 { -(frac_digits as i16) } else { i16::MIN };
            let (buf, exp) = format_exact(decoded, &mut buf[..maxlen], limit);
            if exp <= limit {
                // the restriction couldn't been met, so this should render like zero no matter
                // `exp` was. this does not include the case that the restriction has been met
                // only after the final rounding-up; it's a regular case with `exp = limit + 1`.
                debug_assert_eq!(buf.len(), 0);
                if frac_digits > 0 {
                    // [0.][0000]
                    parts[0] = MaybeUninit::new(Part::Copy(b"0."));
                    parts[1] = MaybeUninit::new(Part::Zero(frac_digits));
                    Formatted {
                        sign,
                        // SAFETY: we just initialized the elements `..2`.
                        parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..2]) },
                    }
                } else {
                    parts[0] = MaybeUninit::new(Part::Copy(b"0"));
                    Formatted {
                        sign,
                        // SAFETY: we just initialized the elements `..1`.
                        parts: unsafe { MaybeUninit::slice_assume_init_ref(&parts[..1]) },
                    }
                }
            } else {
                Formatted { sign, parts: digits_to_dec_str(buf, exp, frac_digits, parts) }
            }
        }
    }
}
