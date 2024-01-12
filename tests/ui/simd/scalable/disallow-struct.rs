#![feature(repr_simd, repr_scalable)]

#[repr(simd, scalable(4))]
pub struct ScalableSimdFloat {
    _ty: [f32]
}

pub struct Invalid {
    x: ScalableSimdFloat, //~ ERROR E0800
    last: i32,
}

#[repr(transparent)]
struct Wrap(ScalableSimdFloat); //~ ERROR E0800

fn main() {
}
