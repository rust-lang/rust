use test::{Bencher, black_box};

use super::corpora;

#[bench]
fn chars_advance_by_1000(b: &mut Bencher) {
    b.iter(|| black_box(corpora::ru::LARGE).chars().advance_by(1000));
}

#[bench]
fn chars_advance_by_0010(b: &mut Bencher) {
    b.iter(|| black_box(corpora::ru::LARGE).chars().advance_by(10));
}

#[bench]
fn chars_advance_by_0001(b: &mut Bencher) {
    b.iter(|| black_box(corpora::ru::LARGE).chars().advance_by(1));
}
