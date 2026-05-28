//@ aux-build:const_evaluatable_lib.rs
//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
extern crate const_evaluatable_lib;

fn user<T>() where [u8; std::mem::size_of::<T>() - 1]: Sized {
    assert_eq!(const_evaluatable_lib::test1::<T>(), [0; std::mem::size_of::<T>() - 1]);
}

fn main() {
    assert_eq!(const_evaluatable_lib::test1::<u32>(), [0; 3]);
    user::<u32>();
    user::<u64>();
}
