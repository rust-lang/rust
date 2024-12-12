use core::f16::consts::PI;

use test::{Bencher, black_box};

#[bench]
fn div_euclid_small(b: &mut Bencher) {
    b.iter(|| black_box(20.12f16).div_euclid(black_box(PI)));
}

#[bench]
fn div_euclid_large(b: &mut Bencher) {
    b.iter(|| black_box(50000.0f16).div_euclid(black_box(PI)));
}
