use core::num::flt2dec::strategy::grisu::*;
use std::mem::MaybeUninit;

use super::super::*;

pub fn decode_finite<T: DecodableFloat>(v: T) -> Decoded {
    match decode(v).1 {
        FullDecoded::Finite(decoded) => decoded,
        full_decoded => panic!("expected finite, got {full_decoded:?} instead"),
    }
}

#[bench]
fn bench_small_shortest(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [MaybeUninit::new(0); MAX_SIG_DIGITS];
    b.iter(|| {
        format_shortest(black_box(&decoded), &mut buf);
    });
}

#[bench]
fn bench_big_shortest(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [MaybeUninit::new(0); MAX_SIG_DIGITS];
    b.iter(|| {
        format_shortest(black_box(&decoded), &mut buf);
    });
}

#[bench]
fn bench_small_exact_3(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [MaybeUninit::new(0); 3];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_big_exact_3(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [MaybeUninit::new(0); 3];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_small_exact_12(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [MaybeUninit::new(0); 12];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_big_exact_12(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [MaybeUninit::new(0); 12];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_small_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [MaybeUninit::new(0); 1024];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_big_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [MaybeUninit::new(0); 1024];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_one_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(1.0);
    let mut buf = [MaybeUninit::new(0); 1024];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_trailing_zero_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(250.000000000000000000000000);
    let mut buf = [MaybeUninit::new(0); 1024];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}

#[bench]
fn bench_halfway_point_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(1.00000000000000011102230246251565404236316680908203125);
    let mut buf = [MaybeUninit::new(0); 1024];
    b.iter(|| {
        format_exact(black_box(&decoded), &mut buf, i16::MIN);
    });
}
