use core::f32::consts::PI;

use test::{Bencher, black_box};

#[bench]
fn div_euclid_small(b: &mut Bencher) {
    b.iter(|| black_box(130.12345f32).div_euclid(black_box(PI)));
}

#[bench]
fn div_euclid_medium(b: &mut Bencher) {
    b.iter(|| black_box(1.123e7f32).div_euclid(black_box(PI)));
}

#[bench]
fn div_euclid_large(b: &mut Bencher) {
    b.iter(|| black_box(1.123e32f32).div_euclid(black_box(PI)));
}
