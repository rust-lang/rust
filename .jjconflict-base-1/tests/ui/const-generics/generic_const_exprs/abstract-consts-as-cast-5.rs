#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const N: u8>(a: [(); N as usize]) {
    bar::<{ N as usize as usize }>();
    //~^ error: unconstrained generic constant
}

fn bar<const N: usize>() {}

fn main() {}
