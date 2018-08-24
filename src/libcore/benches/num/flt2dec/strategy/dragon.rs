use std::{i16, f64};
use super::super::*;
use core::num::flt2dec::strategy::dragon::*;
use test::Bencher;

#[bench]
fn bench_small_shortest(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; MAX_SIG_DIGITS];
    b.iter(|| format_shortest(&decoded, &mut buf));
}

#[bench]
fn bench_big_shortest(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; MAX_SIG_DIGITS];
    b.iter(|| format_shortest(&decoded, &mut buf));
}

#[bench]
fn bench_small_exact_3(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; 3];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_big_exact_3(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; 3];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_small_exact_12(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; 12];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_big_exact_12(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; 12];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_small_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; 1024];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_big_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; 1024];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}
