use core::num::flt2dec::SHORT_DIGITS_MAX;
use core::num::flt2dec::decoder::decode_f64;
use core::num::flt2dec::strategy::dragon::*;
use std::mem::MaybeUninit;

use test::{Bencher, black_box};

#[bench]
fn bench_small_short(b: &mut Bencher) {
    let decoded = decode_f64(3.141592);
    let mut buf = [MaybeUninit::new(0); SHORT_DIGITS_MAX];
    b.iter(|| {
        format_short(black_box(&decoded), &mut buf);
    });
}

#[bench]
fn bench_big_short(b: &mut Bencher) {
    let decoded = decode_f64(f64::MAX);
    let mut buf = [MaybeUninit::new(0); SHORT_DIGITS_MAX];
    b.iter(|| {
        format_short(black_box(&decoded), &mut buf);
    });
}

#[bench]
fn bench_small_fixed_3(b: &mut Bencher) {
    let decoded = decode_f64(3.141592);
    let mut buf = [MaybeUninit::new(0); 3];
    b.iter(|| {
        format_fixed(black_box(&decoded), &mut buf, isize::MIN);
    });
}

#[bench]
fn bench_big_fixed_3(b: &mut Bencher) {
    let decoded = decode_f64(f64::MAX);
    let mut buf = [MaybeUninit::new(0); 3];
    b.iter(|| {
        format_fixed(black_box(&decoded), &mut buf, isize::MIN);
    });
}

#[bench]
fn bench_small_fixed_12(b: &mut Bencher) {
    let decoded = decode_f64(3.141592);
    let mut buf = [MaybeUninit::new(0); 12];
    b.iter(|| {
        format_fixed(black_box(&decoded), &mut buf, isize::MIN);
    });
}

#[bench]
fn bench_big_fixed_12(b: &mut Bencher) {
    let decoded = decode_f64(f64::MAX);
    let mut buf = [MaybeUninit::new(0); 12];
    b.iter(|| {
        format_fixed(black_box(&decoded), &mut buf, isize::MIN);
    });
}

#[bench]
fn bench_small_fixed_inf(b: &mut Bencher) {
    let decoded = decode_f64(3.141592);
    let mut buf = [MaybeUninit::new(0); 1024];
    b.iter(|| {
        format_fixed(black_box(&decoded), &mut buf, isize::MIN);
    });
}

#[bench]
fn bench_big_fixed_inf(b: &mut Bencher) {
    let decoded = decode_f64(f64::MAX);
    let mut buf = [MaybeUninit::new(0); 1024];
    b.iter(|| {
        format_fixed(black_box(&decoded), &mut buf, isize::MIN);
    });
}
