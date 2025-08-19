#![feature(float_gamma, portable_simd, core_intrinsics)]
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::hint::black_box;

fn ldexp(a: f64, b: i32) -> f64 {
    extern "C" {
        fn ldexp(x: f64, n: i32) -> f64;
    }
    unsafe { ldexp(a, b) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Neg = 1,
    Pos = 0,
}
use Sign::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NaNKind {
    Quiet = 1,
    Signaling = 0,
}
use NaNKind::*;

#[track_caller]
fn check_all_outcomes<T: Eq + Hash + fmt::Display>(expected: HashSet<T>, generate: impl Fn() -> T) {
    let mut seen = HashSet::new();
    // Let's give it sixteen times as many tries as we are expecting values.
    let tries = expected.len() * 16;
    for _ in 0..tries {
        let val = generate();
        assert!(expected.contains(&val), "got an unexpected value: {val}");
        seen.insert(val);
    }
    // Let's see if we saw them all.
    for val in expected {
        if !seen.contains(&val) {
            panic!("did not get value that should be possible: {val}");
        }
    }
}

// -- f32 support
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct F32(u32);

impl From<f32> for F32 {
    fn from(x: f32) -> Self {
        F32(x.to_bits())
    }
}

/// Returns a value that is `ones` many 1-bits.
fn u32_ones(ones: u32) -> u32 {
    assert!(ones <= 32);
    if ones == 0 {
        // `>>` by 32 doesn't actually shift. So inconsistent :(
        return 0;
    }
    u32::MAX >> (32 - ones)
}

const F32_SIGN_BIT: u32 = 32 - 1; // position of the sign bit
const F32_EXP: u32 = 8; // 8 bits of exponent
const F32_MANTISSA: u32 = F32_SIGN_BIT - F32_EXP;
const F32_NAN_PAYLOAD: u32 = F32_MANTISSA - 1;

impl fmt::Display for F32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Alaways show raw bits.
        write!(f, "0x{:08x} ", self.0)?;
        // Also show nice version.
        let val = self.0;
        let sign = val >> F32_SIGN_BIT;
        let val = val & u32_ones(F32_SIGN_BIT); // mask away sign
        let exp = val >> F32_MANTISSA;
        let mantissa = val & u32_ones(F32_MANTISSA);
        if exp == u32_ones(F32_EXP) {
            // A NaN! Special printing.
            let sign = if sign != 0 { Neg } else { Pos };
            let quiet = if (mantissa >> F32_NAN_PAYLOAD) != 0 { Quiet } else { Signaling };
            let payload = mantissa & u32_ones(F32_NAN_PAYLOAD);
            write!(f, "(NaN: {:?}, {:?}, payload = {:#x})", sign, quiet, payload)
        } else {
            // Normal float value.
            write!(f, "({})", f32::from_bits(self.0))
        }
    }
}

impl F32 {
    fn nan(sign: Sign, kind: NaNKind, payload: u32) -> Self {
        // Either the quiet bit must be set of the payload must be non-0;
        // otherwise this is not a NaN but an infinity.
        assert!(kind == Quiet || payload != 0);
        // Payload must fit in 22 bits.
        assert!(payload < (1 << F32_NAN_PAYLOAD));
        // Concatenate the bits (with a 22bit payload).
        // Pattern: [negative] ++ [1]^8 ++ [quiet] ++ [payload]
        let val = ((sign as u32) << F32_SIGN_BIT)
            | (u32_ones(F32_EXP) << F32_MANTISSA)
            | ((kind as u32) << F32_NAN_PAYLOAD)
            | payload;
        // Sanity check.
        assert!(f32::from_bits(val).is_nan());
        // Done!
        F32(val)
    }

    fn as_f32(self) -> f32 {
        black_box(f32::from_bits(self.0))
    }
}

// -- f64 support
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct F64(u64);

impl From<f64> for F64 {
    fn from(x: f64) -> Self {
        F64(x.to_bits())
    }
}

/// Returns a value that is `ones` many 1-bits.
fn u64_ones(ones: u32) -> u64 {
    assert!(ones <= 64);
    if ones == 0 {
        // `>>` by 32 doesn't actually shift. So inconsistent :(
        return 0;
    }
    u64::MAX >> (64 - ones)
}

const F64_SIGN_BIT: u32 = 64 - 1; // position of the sign bit
const F64_EXP: u32 = 11; // 11 bits of exponent
const F64_MANTISSA: u32 = F64_SIGN_BIT - F64_EXP;
const F64_NAN_PAYLOAD: u32 = F64_MANTISSA - 1;

impl fmt::Display for F64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Alaways show raw bits.
        write!(f, "0x{:08x} ", self.0)?;
        // Also show nice version.
        let val = self.0;
        let sign = val >> F64_SIGN_BIT;
        let val = val & u64_ones(F64_SIGN_BIT); // mask away sign
        let exp = val >> F64_MANTISSA;
        let mantissa = val & u64_ones(F64_MANTISSA);
        if exp == u64_ones(F64_EXP) {
            // A NaN! Special printing.
            let sign = if sign != 0 { Neg } else { Pos };
            let quiet = if (mantissa >> F64_NAN_PAYLOAD) != 0 { Quiet } else { Signaling };
            let payload = mantissa & u64_ones(F64_NAN_PAYLOAD);
            write!(f, "(NaN: {:?}, {:?}, payload = {:#x})", sign, quiet, payload)
        } else {
            // Normal float value.
            write!(f, "({})", f64::from_bits(self.0))
        }
    }
}

impl F64 {
    fn nan(sign: Sign, kind: NaNKind, payload: u64) -> Self {
        // Either the quiet bit must be set of the payload must be non-0;
        // otherwise this is not a NaN but an infinity.
        assert!(kind == Quiet || payload != 0);
        // Payload must fit in 52 bits.
        assert!(payload < (1 << F64_NAN_PAYLOAD));
        // Concatenate the bits (with a 52bit payload).
        // Pattern: [negative] ++ [1]^11 ++ [quiet] ++ [payload]
        let val = ((sign as u64) << F64_SIGN_BIT)
            | (u64_ones(F64_EXP) << F64_MANTISSA)
            | ((kind as u64) << F64_NAN_PAYLOAD)
            | payload;
        // Sanity check.
        assert!(f64::from_bits(val).is_nan());
        // Done!
        F64(val)
    }

    fn as_f64(self) -> f64 {
        black_box(f64::from_bits(self.0))
    }
}

// -- actual tests

fn test_f32() {
    // Freshly generated NaNs can have either sign.
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(0.0 / black_box(0.0)),
    );
    // When there are NaN inputs, their payload can be propagated, with any sign.
    let all1_payload = u32_ones(22);
    let all1 = F32::nan(Pos, Quiet, all1_payload).as_f32();
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, all1_payload),
            F32::nan(Neg, Quiet, all1_payload),
        ]),
        || F32::from(0.0 + all1),
    );
    // When there are two NaN inputs, the output can be either one, or the preferred NaN.
    let just1 = F32::nan(Neg, Quiet, 1).as_f32();
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, 1),
            F32::nan(Neg, Quiet, 1),
            F32::nan(Pos, Quiet, all1_payload),
            F32::nan(Neg, Quiet, all1_payload),
        ]),
        || F32::from(just1 - all1),
    );
    // When there are *signaling* NaN inputs, they might be quieted or not.
    let all1_snan = F32::nan(Pos, Signaling, all1_payload).as_f32();
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, all1_payload),
            F32::nan(Neg, Quiet, all1_payload),
            F32::nan(Pos, Signaling, all1_payload),
            F32::nan(Neg, Signaling, all1_payload),
        ]),
        || F32::from(0.0 * all1_snan),
    );
    // Mix signaling and non-signaling NaN.
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, 1),
            F32::nan(Neg, Quiet, 1),
            F32::nan(Pos, Quiet, all1_payload),
            F32::nan(Neg, Quiet, all1_payload),
            F32::nan(Pos, Signaling, all1_payload),
            F32::nan(Neg, Signaling, all1_payload),
        ]),
        || F32::from(just1 % all1_snan),
    );

    // Unary `-` must preserve payloads exactly.
    check_all_outcomes(HashSet::from_iter([F32::nan(Neg, Quiet, all1_payload)]), || {
        F32::from(-all1)
    });
    check_all_outcomes(HashSet::from_iter([F32::nan(Neg, Signaling, all1_payload)]), || {
        F32::from(-all1_snan)
    });

    // Intrinsics
    let nan = F32::nan(Neg, Quiet, 0).as_f32();
    let snan = F32::nan(Neg, Signaling, 1).as_f32();
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(f32::min(nan, nan)),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(nan.floor()),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(nan.sin()),
    );
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, 1),
            F32::nan(Neg, Quiet, 1),
            F32::nan(Pos, Quiet, 2),
            F32::nan(Neg, Quiet, 2),
            F32::nan(Pos, Quiet, all1_payload),
            F32::nan(Neg, Quiet, all1_payload),
            F32::nan(Pos, Signaling, all1_payload),
            F32::nan(Neg, Signaling, all1_payload),
        ]),
        || F32::from(just1.mul_add(F32::nan(Neg, Quiet, 2).as_f32(), all1_snan)),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(nan.powf(nan)),
    );
    check_all_outcomes(
        HashSet::from_iter([1.0f32.into()]),
        || F32::from(1.0f32.powf(nan)), // special `pow` rule
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(nan.powi(1)),
    );

    // libm functions
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(nan.sinh()),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(nan.atan2(nan)),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(nan.ln_gamma().0),
    );
    check_all_outcomes(
        HashSet::from_iter([
            F32::from(1.0),
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, 1),
            F32::nan(Neg, Quiet, 1),
            F32::nan(Pos, Signaling, 1),
            F32::nan(Neg, Signaling, 1),
        ]),
        || F32::from(snan.powf(0.0)),
    );
}

fn test_f64() {
    // Freshly generated NaNs can have either sign.
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(0.0 / black_box(0.0)),
    );
    // When there are NaN inputs, their payload can be propagated, with any sign.
    let all1_payload = u64_ones(51);
    let all1 = F64::nan(Pos, Quiet, all1_payload).as_f64();
    check_all_outcomes(
        HashSet::from_iter([
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, all1_payload),
            F64::nan(Neg, Quiet, all1_payload),
        ]),
        || F64::from(0.0 + all1),
    );
    // When there are two NaN inputs, the output can be either one, or the preferred NaN.
    let just1 = F64::nan(Neg, Quiet, 1).as_f64();
    check_all_outcomes(
        HashSet::from_iter([
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, 1),
            F64::nan(Neg, Quiet, 1),
            F64::nan(Pos, Quiet, all1_payload),
            F64::nan(Neg, Quiet, all1_payload),
        ]),
        || F64::from(just1 - all1),
    );
    // When there are *signaling* NaN inputs, they might be quieted or not.
    let all1_snan = F64::nan(Pos, Signaling, all1_payload).as_f64();
    check_all_outcomes(
        HashSet::from_iter([
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, all1_payload),
            F64::nan(Neg, Quiet, all1_payload),
            F64::nan(Pos, Signaling, all1_payload),
            F64::nan(Neg, Signaling, all1_payload),
        ]),
        || F64::from(0.0 * all1_snan),
    );
    // Mix signaling and non-signaling NaN.
    check_all_outcomes(
        HashSet::from_iter([
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, 1),
            F64::nan(Neg, Quiet, 1),
            F64::nan(Pos, Quiet, all1_payload),
            F64::nan(Neg, Quiet, all1_payload),
            F64::nan(Pos, Signaling, all1_payload),
            F64::nan(Neg, Signaling, all1_payload),
        ]),
        || F64::from(just1 % all1_snan),
    );

    // Intrinsics
    let nan = F64::nan(Neg, Quiet, 0).as_f64();
    let snan = F64::nan(Neg, Signaling, 1).as_f64();
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(f64::min(nan, nan)),
    );
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(nan.floor()),
    );
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(nan.sin()),
    );
    check_all_outcomes(
        HashSet::from_iter([
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, 1),
            F64::nan(Neg, Quiet, 1),
            F64::nan(Pos, Quiet, 2),
            F64::nan(Neg, Quiet, 2),
            F64::nan(Pos, Quiet, all1_payload),
            F64::nan(Neg, Quiet, all1_payload),
            F64::nan(Pos, Signaling, all1_payload),
            F64::nan(Neg, Signaling, all1_payload),
        ]),
        || F64::from(just1.mul_add(F64::nan(Neg, Quiet, 2).as_f64(), all1_snan)),
    );
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(nan.powf(nan)),
    );
    check_all_outcomes(
        HashSet::from_iter([1.0f64.into()]),
        || F64::from(1.0f64.powf(nan)), // special `pow` rule
    );
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(nan.powi(1)),
    );

    // libm functions
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(nan.sinh()),
    );
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(nan.atan2(nan)),
    );
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(ldexp(nan, 1)),
    );
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(nan.ln_gamma().0),
    );
    check_all_outcomes(
        HashSet::from_iter([
            F64::from(1.0),
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, 1),
            F64::nan(Neg, Quiet, 1),
            F64::nan(Pos, Signaling, 1),
            F64::nan(Neg, Signaling, 1),
        ]),
        || F64::from(snan.powf(0.0)),
    );
}

fn test_casts() {
    let all1_payload_32 = u32_ones(22);
    let all1_payload_64 = u64_ones(51);
    let left1_payload_64 = (all1_payload_32 as u64) << (51 - 22);

    // 64-to-32
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(F64::nan(Pos, Quiet, 0).as_f64() as f32),
    );
    // The preferred payload is always a possibility.
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, all1_payload_32),
            F32::nan(Neg, Quiet, all1_payload_32),
        ]),
        || F32::from(F64::nan(Pos, Quiet, all1_payload_64).as_f64() as f32),
    );
    // If the input is signaling, then the output *may* also be signaling.
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, all1_payload_32),
            F32::nan(Neg, Quiet, all1_payload_32),
            F32::nan(Pos, Signaling, all1_payload_32),
            F32::nan(Neg, Signaling, all1_payload_32),
        ]),
        || F32::from(F64::nan(Pos, Signaling, all1_payload_64).as_f64() as f32),
    );
    // Check that the low bits are gone (not the high bits).
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(F64::nan(Pos, Quiet, 1).as_f64() as f32),
    );
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            F32::nan(Pos, Quiet, 1),
            F32::nan(Neg, Quiet, 1),
        ]),
        || F32::from(F64::nan(Pos, Quiet, 1 << (51 - 22)).as_f64() as f32),
    );
    check_all_outcomes(
        HashSet::from_iter([
            F32::nan(Pos, Quiet, 0),
            F32::nan(Neg, Quiet, 0),
            // The `1` payload becomes `0`, and the `0` payload cannot be signaling,
            // so these are the only options.
        ]),
        || F32::from(F64::nan(Pos, Signaling, 1).as_f64() as f32),
    );

    // 32-to-64
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(F32::nan(Pos, Quiet, 0).as_f32() as f64),
    );
    // The preferred payload is always a possibility.
    // Also checks that 0s are added on the right.
    check_all_outcomes(
        HashSet::from_iter([
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, left1_payload_64),
            F64::nan(Neg, Quiet, left1_payload_64),
        ]),
        || F64::from(F32::nan(Pos, Quiet, all1_payload_32).as_f32() as f64),
    );
    // If the input is signaling, then the output *may* also be signaling.
    check_all_outcomes(
        HashSet::from_iter([
            F64::nan(Pos, Quiet, 0),
            F64::nan(Neg, Quiet, 0),
            F64::nan(Pos, Quiet, left1_payload_64),
            F64::nan(Neg, Quiet, left1_payload_64),
            F64::nan(Pos, Signaling, left1_payload_64),
            F64::nan(Neg, Signaling, left1_payload_64),
        ]),
        || F64::from(F32::nan(Pos, Signaling, all1_payload_32).as_f32() as f64),
    );
}

fn test_simd() {
    use std::intrinsics::simd::*;
    use std::simd::*;

    let nan = F32::nan(Neg, Quiet, 0).as_f32();
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(unsafe { simd_div(f32x4::splat(0.0), f32x4::splat(0.0)) }[0]),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(unsafe { simd_fmin(f32x4::splat(nan), f32x4::splat(nan)) }[0]),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(unsafe { simd_fmax(f32x4::splat(nan), f32x4::splat(nan)) }[0]),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || {
            F32::from(
                unsafe { simd_fma(f32x4::splat(nan), f32x4::splat(nan), f32x4::splat(nan)) }[0],
            )
        },
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(unsafe { simd_reduce_add_ordered::<_, f32>(f32x4::splat(nan), nan) }),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(unsafe { simd_reduce_max::<_, f32>(f32x4::splat(nan)) }),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(unsafe { simd_fsqrt(f32x4::splat(nan)) }[0]),
    );
    check_all_outcomes(
        HashSet::from_iter([F32::nan(Pos, Quiet, 0), F32::nan(Neg, Quiet, 0)]),
        || F32::from(unsafe { simd_ceil(f32x4::splat(nan)) }[0]),
    );

    // Casts
    check_all_outcomes(
        HashSet::from_iter([F64::nan(Pos, Quiet, 0), F64::nan(Neg, Quiet, 0)]),
        || F64::from(unsafe { simd_cast::<f32x4, f64x4>(f32x4::splat(nan)) }[0]),
    );
}

fn main() {
    // Check our constants against std, just to be sure.
    // We add 1 since our numbers are the number of bits stored
    // to represent the value, and std has the precision of the value,
    // which is one more due to the implicit leading 1.
    assert_eq!(F32_MANTISSA + 1, f32::MANTISSA_DIGITS);
    assert_eq!(F64_MANTISSA + 1, f64::MANTISSA_DIGITS);

    test_f32();
    test_f64();
    test_casts();
    test_simd();
}
