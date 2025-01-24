use core::f64::consts::PI;

use test::{Bencher, black_box};

#[bench]
fn div_euclid_small(b: &mut Bencher) {
    b.iter(|| black_box(1234.1234578f64).div_euclid(black_box(PI)));
}

#[bench]
fn div_euclid_medium(b: &mut Bencher) {
    b.iter(|| black_box(1.123e15f64).div_euclid(black_box(PI)));
}

#[bench]
fn div_euclid_large(b: &mut Bencher) {
    b.iter(|| black_box(1.123e300f64).div_euclid(black_box(PI)));
}
