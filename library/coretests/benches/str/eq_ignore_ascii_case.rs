use test::{Bencher, black_box};

use super::corpora::*;

#[bench]
fn bench_str_under_8_bytes_eq(b: &mut Bencher) {
    let s = black_box("foo");
    let other = black_box("foo");
    b.iter(|| assert!(s.eq_ignore_ascii_case(other)))
}

#[bench]
fn bench_str_of_8_bytes_eq(b: &mut Bencher) {
    let s = black_box(en::TINY);
    let other = black_box(en::TINY);
    b.iter(|| assert!(s.eq_ignore_ascii_case(other)))
}

#[bench]
fn bench_str_17_bytes_eq(b: &mut Bencher) {
    let s = black_box(&en::SMALL[..17]);
    let other = black_box(&en::SMALL[..17]);
    b.iter(|| assert!(s.eq_ignore_ascii_case(other)))
}

#[bench]
fn bench_str_31_bytes_eq(b: &mut Bencher) {
    let s = black_box(&en::SMALL[..31]);
    let other = black_box(&en::SMALL[..31]);
    b.iter(|| assert!(s.eq_ignore_ascii_case(other)))
}

#[bench]
fn bench_medium_str_eq(b: &mut Bencher) {
    let s = black_box(en::MEDIUM);
    let other = black_box(en::MEDIUM);
    b.iter(|| assert!(s.eq_ignore_ascii_case(other)))
}

#[bench]
fn bench_large_str_eq(b: &mut Bencher) {
    let s = black_box(en::LARGE);
    let other = black_box(en::LARGE);
    b.iter(|| assert!(s.eq_ignore_ascii_case(other)))
}
