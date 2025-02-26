//@ run-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(stable_features)]
#![allow(dropping_copy_types)]

// Test parsing binary operators after macro invocations.


#![feature(macro_rules)]

macro_rules! id {
    ($e: expr) => { $e }
}

fn foo() {
    id!(1) + 1;
    id![1] - 1;
    id!(1) * 1;
    id![1] / 1;
    id!(1) % 1;

    id!(1) & 1;
    id![1] | 1;
    id!(1) ^ 1;

    let mut x = 1;
    id![x] = 2;
    id!(x) += 1;

    id!(1f64).clone();

    id!([1, 2, 3])[1];
    id![drop](1);

    id!(true) && true;
    id![true] || true;
}

fn main() {}
