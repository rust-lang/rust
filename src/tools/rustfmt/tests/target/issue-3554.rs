#![feature(const_generics)]

pub struct S<const N: usize>;
impl S<{ 0 }> {}
