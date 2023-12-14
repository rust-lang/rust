#![feature(repr_simd)]

#[repr(simd, scalable(16))] //~ error: Scalable SIMD types are experimental
struct Foo {
    _ty: [i8],
}

fn main() {}
