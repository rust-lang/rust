// aux-build:const_evaluatable_lib.rs
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]
extern crate const_evaluatable_lib;

fn user<T>() {
    let _ = const_evaluatable_lib::test1::<T>();
    //~^ ERROR constant expression depends
    //~| ERROR constant expression depends
    //~| ERROR constant expression depends
    //~| ERROR constant expression depends
}

fn main() {}
