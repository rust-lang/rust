#![feature(repr_simd)]

#[repr(simd, scalable(16))] //~ ERROR: scalable vector types are experimental
struct Foo {
    _ty: [i8],
}

fn main() {}
