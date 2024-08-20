//@ known-bug: rust-lang/rust#129150
//@ only-x86_64
use std::arch::x86_64::_mm_blend_ps;

pub fn main() {
     _mm_blend_ps(1, 2, &const {} );
}
