#![feature(repr_simd)]

#[repr(simd, scalable(16))] //~ ERROR: scalable vector types are experimental
struct Foo { //~ ERROR: SIMD vector's only field must be an array
    _ty: [i8],
}

fn main() {}
