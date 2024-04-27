//@ aux-build:const_evaluatable_lib.rs
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
extern crate const_evaluatable_lib;

fn user<T>() {
    let _ = const_evaluatable_lib::test1::<T>();
    //~^ ERROR unconstrained generic constant
    //~| ERROR unconstrained generic constant
    //~| ERROR unconstrained generic constant
    //~| ERROR unconstrained generic constant
}

fn main() {}
