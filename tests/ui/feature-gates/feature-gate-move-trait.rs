//@ run-pass
#![feature(move_trait)]
#![feature(negative_impls)]
#![feature(min_specialization)]

use std::marker::Move;

struct A;
struct B;
impl !Move for B {}

fn main() {
    let a = A;
    let _new_a = a;

    let b = B;
    let _new_b = b;
}
