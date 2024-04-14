//@ known-bug: #123077
//@ only-x86_64
use std::arch::x86_64::{__m128, _mm_blend_ps};

pub fn sse41_blend_noinline( ) -> __m128 {
    let f = { |x, y| unsafe {
        _mm_blend_ps(x, y, { |x, y| unsafe })
    }};
    f(x, y)
}

pub fn main() {}
