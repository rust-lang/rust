#![feature(repr_simd, repr_scalable)]

#[repr(simd, scalable(4))]
pub struct ScalableSimdFloat {
    _ty: [f32]
}

pub union Invalid {
    x: ScalableSimdFloat,
    //~^ ERROR E0740
    //~^^ ERROR E0800
    other: i32,
}

fn main() {
}
