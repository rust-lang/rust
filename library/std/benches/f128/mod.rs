use core::f128::consts::PI;

use test::{Bencher, black_box};

#[bench]
fn div_euclid_small(b: &mut Bencher) {
    b.iter(|| black_box(10000.12345f128).div_euclid(black_box(PI)));
}

#[bench]
fn div_euclid_medium(b: &mut Bencher) {
    b.iter(|| black_box(1.123e30f128).div_euclid(black_box(PI)));
}

#[bench]
fn div_euclid_large(b: &mut Bencher) {
    b.iter(|| black_box(1.123e4000f128).div_euclid(black_box(PI)));
}
