// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]

#![feature(specialization)]

// Regression test for ICE when combining specialized associated types and type
// aliases

trait Id_ {
    type Out;
}

type Id<T> = <T as Id_>::Out;

impl<T> Id_ for T {
    default type Out = T;
}

fn test_proection() {
    let x: Id<bool> = panic!();
}

fn main() {

}
