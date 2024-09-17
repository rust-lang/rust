//@ run-pass
// A regression test for https://github.com/rust-lang/rust/issues/130402
// Our SIMD representation did not combine correctly with the repr(align) attribute,
// and this will remain a concern regardless of what we do with SIMD types.
#![feature(repr_simd)]
use std::mem::{size_of, align_of};

#[repr(simd, align(64))]
struct IntelsIdeaOfWhatAvx512Means([u8; 32]);

#[repr(transparent)]
struct DesignValidation(IntelsIdeaOfWhatAvx512Means);

fn main() {
    assert_eq!(64, size_of::<IntelsIdeaOfWhatAvx512Means>());
    assert_eq!(64, align_of::<IntelsIdeaOfWhatAvx512Means>());
    assert_eq!(64, size_of::<DesignValidation>());
    assert_eq!(64, align_of::<DesignValidation>());
}
