//@ run-rustfix
#![allow(dead_code)]

fn mutate(_y: &mut i32) {}

fn foo(&x: &i32) {
    mutate(&mut x);
    //~^ ERROR cannot borrow `x` as mutable
}

fn main() {}
