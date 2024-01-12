#![feature(repr_simd, repr_scalable)]

#[repr(simd, scalable(4))]
pub struct ScalableSimdFloat {
    _ty: [f32]
}

fn main() {
    let x: [ScalableSimdFloat; 2]; //~ ERROR E0277
}
