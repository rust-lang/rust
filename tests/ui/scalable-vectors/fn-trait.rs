#![allow(internal_features)]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(4)]
pub struct ScalableSimdFloat(f32);

unsafe fn test<T>(f: T)
where
    T: Fn(ScalableSimdFloat), //~ ERROR: scalable vectors cannot be tuple fields
{
}

fn main() {}
