use std::str;

use test::{Bencher, black_box};

mod char_count;
mod corpora;
mod debug;
mod iter;

#[bench]
fn str_validate_emoji(b: &mut Bencher) {
    b.iter(|| str::from_utf8(black_box(corpora::emoji::LARGE.as_bytes())));
}
