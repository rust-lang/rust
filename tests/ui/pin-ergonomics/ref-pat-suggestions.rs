//@ run-rustfix

#![feature(pin_ergonomics)]
#![allow(unused)]

fn pin_mut_param(&pin mut x: i32) {}
//~^ ERROR mismatched types

fn pin_const_param(&pin const x: i32) {}
//~^ ERROR mismatched types

fn main() {}
