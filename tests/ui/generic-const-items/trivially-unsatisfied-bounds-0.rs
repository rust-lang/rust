#![feature(generic_const_items, trivial_bounds)]
#![allow(incomplete_features)]

// Ensure that we check if trivial bounds on const items hold or not.

const UNUSABLE: () = () //~ ERROR evaluation of constant value failed
where
    String: Copy;

fn main() {
    let _ = UNUSABLE; //~ ERROR the trait bound `String: Copy` is not satisfied
}
