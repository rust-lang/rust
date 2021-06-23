#![feature(const_evaluatable_checked, const_generics, const_generics_defaults)]
#![allow(incomplete_features)]

pub struct Bar<const N: usize, const M: usize = { N + 1 }>;
pub fn foo<const N1: usize>() -> Bar<N1> { 
    loop {} 
}
//~^ error: unconstrained generic constant

fn main() {}
