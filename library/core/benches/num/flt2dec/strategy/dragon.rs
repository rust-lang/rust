use core::num::flt2dec::strategy::dragon::*;
use std::mem::MaybeUninit;

use super::super::*;

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
