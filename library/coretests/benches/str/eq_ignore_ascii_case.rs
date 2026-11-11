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

#[bench]
fn bench_medium_str_early_mismatch(b: &mut Bencher) {
    // Differs in the first byte, exercising the early-return path of the
    // chunked comparison.
    let other_owned = ["_", &en::MEDIUM[1..]].concat();
    let s = black_box(en::MEDIUM);
    let other = black_box(&*other_owned);
    b.iter(|| assert!(!s.eq_ignore_ascii_case(other)))
}

#[bench]
fn bench_medium_str_tail_mismatch(b: &mut Bencher) {
    // Differs in the last byte, exercising the `last_chunk` tail handling of
    // the chunked comparison.
    let other_owned = [&en::MEDIUM[..en::MEDIUM.len() - 1], "_"].concat();
    let s = black_box(en::MEDIUM);
    let other = black_box(&*other_owned);
    b.iter(|| assert!(!s.eq_ignore_ascii_case(other)))
}
