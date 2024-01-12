#![feature(repr_simd, repr_scalable)]

#[repr(simd, scalable(4))]
pub struct ScalableSimdFloat {
    _ty: [f32]
}

pub enum Invalid {
    Scalable(ScalableSimdFloat), //~ ERROR E0799
    Int(i32),
}

fn main() {
}
