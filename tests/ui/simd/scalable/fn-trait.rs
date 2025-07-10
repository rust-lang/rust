#![feature(repr_simd, repr_scalable)]

#[repr(simd, scalable(4))]
pub struct ScalableSimdFloat {
    _ty: [f32],
}

unsafe fn test<T>(f: T)
where
    T: Fn(ScalableSimdFloat), //~ ERROR: scalable vectors cannot be tuple fields
{
}

fn main() {}
