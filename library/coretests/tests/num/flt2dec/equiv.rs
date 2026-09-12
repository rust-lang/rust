#![cfg(not(target_arch = "wasm32"))]

use core::num::imp::flt2dec;
use std::mem::MaybeUninit;
use std::string::String;

use flt2dec::strategy::{dragon, grisu};
use flt2dec::{DecodableFloat, Decoded, FullDecoded, MAX_SIG_DIGITS, decode};
use rand::distr::{Distribution, Uniform};

// Bits 0u16..0x7C00 cover the positive finite-range,
// with 1u16..0x7C00 for non-zero.
const F16_POS_FIN_RANGE: std::ops::Range<u16> = 1..f16::INFINITY.to_bits();

// Bits 0u32..0x7F80_0000 cover the positive finite-range,
// with 1u32..0x7F80_0000 for non-zero.
const F32_POS_FIN_RANGE: std::ops::Range<u32> = 1..f32::INFINITY.to_bits();

// Bits 0u64..0x7FF0_0000_0000_0000 cover the positive finite-range,
// with 1u64..0x7FF0_0000_0000_0000 for non-zero.
const F64_POS_FIN_RANGE: std::ops::Range<u64> = 1..f64::INFINITY.to_bits();

fn decode_finite<T: DecodableFloat>(v: T) -> Decoded {
    match decode(v).1 {
        FullDecoded::Finite(decoded) => decoded,
        full_decoded => panic!("expected finite, got {full_decoded:?} instead"),
    }
}

/// Verifies whether `format_short` of Grisu and Dragon both have the same
/// outcome for n `Decoded` values from `v`. Values which get `None` from Grisu
/// are omitted. The return counts the amount omitted, range 0..n.
fn n_short_equiv<V>(n: usize, mut v: V) -> usize
where
    V: FnMut(usize) -> Decoded,
{
    let mut omitted = 0;
    let mut failed = 0;
    for i in 0..n {
        if (i & 0xf_ffff) == 0 {
            println!("did {i} out of {n}, with {omitted} omitted");
        }
        let dec = v(i);

        let mut grisu_buf = [MaybeUninit::new(b'_'); MAX_SIG_DIGITS];
        let mut dragon_buf = [MaybeUninit::new(b'_'); MAX_SIG_DIGITS];
        if let Some(grisu_return) = grisu::format_short(&dec, &mut grisu_buf) {
            let dragon_return = dragon::format_short(&dec, &mut dragon_buf);
            if grisu_return != dragon_return {
                failed += 1;
                println!(
                    "grisu got ({}, {}) while dragon got ({}, {}) for {:?}",
                    String::from_utf8_lossy(grisu_return.0),
                    grisu_return.1,
                    String::from_utf8_lossy(dragon_return.0),
                    dragon_return.1,
                    dec,
                );
            }
        } else {
            omitted += 1;
        }
    }
    assert_eq!(failed, 0, "number of different outcomes");
    omitted
}

/// Verifies whether `format_fixed` of Grisu and Dragon both have the same
/// outcome for n `Decoded` values from `v`. Values which get `None` from Grisu
/// are omitted. The return counts the amount omitted, range 0..n.
fn n_fixed_equiv<V>(n: usize, bufn: usize, mut v: V) -> usize
where
    V: FnMut(usize) -> Decoded,
{
    const BUF_CAP: usize = 64;
    assert!(bufn <= BUF_CAP, "just increase capacity when needed");
    const RESOLUTION: i16 = i16::MIN; // unlimited

    let mut omitted = 0;
    let mut failed = 0;
    for i in 0..n {
        if (i & 0xf_ffff) == 0 {
            println!("did {i} out of {n}, with {omitted} omitted");
        }
        let dec = v(i);

        let mut grisu_buf = [MaybeUninit::new(b'_'); BUF_CAP];
        let mut dragon_buf = [MaybeUninit::new(b'_'); BUF_CAP];
        if let Some(grisu_return) = grisu::format_fixed(&dec, &mut grisu_buf[..bufn], RESOLUTION) {
            let dragon_return = dragon::format_fixed(&dec, &mut dragon_buf[..bufn], RESOLUTION);
            if grisu_return != dragon_return {
                failed += 1;
                println!(
                    "grisu got ({}, {}) while dragon got ({}, {}) for {:?} with a {}-byte buffer",
                    String::from_utf8_lossy(grisu_return.0),
                    grisu_return.1,
                    String::from_utf8_lossy(dragon_return.0),
                    dragon_return.1,
                    dec,
                    bufn,
                );
            }
        } else {
            omitted += 1;
        }
    }
    assert_eq!(failed, 0, "number of different outcomes");
    omitted
}

#[test]
#[cfg(target_has_reliable_f16)]
fn test_short_f16_random_equiv() {
    // Miri is too slow
    let n = if cfg!(miri) { 10 } else { 10_000 };

    let mut rng = crate::test_rng();
    let u = Uniform::new(F16_POS_FIN_RANGE.start, F16_POS_FIN_RANGE.end).unwrap();
    n_short_equiv(n, |_| {
        let x = f16::from_bits(u.sample(&mut rng));
        decode_finite(x)
    });
}

#[test]
fn test_short_f32_random_equiv() {
    // Miri is too slow
    let n = if cfg!(miri) { 10 } else { 10_000 };

    let mut rng = crate::test_rng();
    let u = Uniform::new(F32_POS_FIN_RANGE.start, F32_POS_FIN_RANGE.end).unwrap();
    n_short_equiv(n, |_| {
        let x = f32::from_bits(u.sample(&mut rng));
        decode_finite(x)
    });
}

#[test]
fn test_short_f64_random_equiv() {
    // Miri is too slow
    let n = if cfg!(miri) { 10 } else { 10_000 };

    let mut rng = crate::test_rng();
    let u = Uniform::new(F64_POS_FIN_RANGE.start, F64_POS_FIN_RANGE.end).unwrap();
    n_short_equiv(n, |_| {
        let x = f64::from_bits(u.sample(&mut rng));
        decode_finite(x)
    });
}

/// Unlike the other float types, `f16` is small enough that these exhaustive tests
/// can run in less than a second so we don't need to ignore it.
#[test]
#[cfg_attr(miri, ignore)] // Miri is to slow
#[cfg(target_has_reliable_f16)]
fn test_short_f16_exhaustive_equiv() {
    let omitted = n_short_equiv(F16_POS_FIN_RANGE.len(), |i: usize| {
        let x = f16::from_bits(F16_POS_FIN_RANGE.start + i as u16);
        decode_finite(x)
    });
    assert_eq!(omitted, 2008, "number of inputs in {F16_POS_FIN_RANGE:?} not covered by Grisu");
}

#[test]
#[ignore] // takes about 40 seconds to run
#[cfg_attr(miri, ignore)] // Miri is to slow
fn test_short_f32_exhaustive_equiv() {
    let omitted = n_short_equiv(F32_POS_FIN_RANGE.len(), |i: usize| {
        let x = f32::from_bits(F32_POS_FIN_RANGE.start + i as u32);
        decode_finite(x)
    });
    assert_eq!(omitted, 17643158, "number of inputs in {F32_POS_FIN_RANGE:?} not covered by Grisu");
}

#[test]
#[cfg(target_has_reliable_f16)]
fn test_fixed_f16_random_equiv() {
    // Miri is too slow
    let n = if cfg!(miri) { 3 } else { 1_000 };

    for bufn in 1..21 {
        let mut rng = crate::test_rng();
        let f16_range = Uniform::new(F16_POS_FIN_RANGE.start, F16_POS_FIN_RANGE.end).unwrap();
        n_fixed_equiv(n, bufn, |_| {
            let x = f16::from_bits(f16_range.sample(&mut rng));
            decode_finite(x)
        });
    }
}

#[test]
fn test_fixed_f32_random_equiv() {
    // Miri is too slow
    let n = if cfg!(miri) { 3 } else { 1_000 };

    for bufn in 1..21 {
        let mut rng = crate::test_rng();
        let f32_range = Uniform::new(F32_POS_FIN_RANGE.start, F32_POS_FIN_RANGE.end).unwrap();
        n_fixed_equiv(n, bufn, |_| {
            let x = f32::from_bits(f32_range.sample(&mut rng));
            decode_finite(x)
        });
    }
}

#[test]
fn test_fixed_f64_random_equiv() {
    // Miri is too slow
    let n = if cfg!(miri) { 2 } else { 1_000 };

    for bufn in 1..21 {
        let mut rng = crate::test_rng();
        let f64_range = Uniform::new(F64_POS_FIN_RANGE.start, F64_POS_FIN_RANGE.end).unwrap();
        n_fixed_equiv(n, bufn, |_| {
            let x = f64::from_bits(f64_range.sample(&mut rng));
            decode_finite(x)
        });
    }
}
