#![feature(generic_const_items)]
#![allow(incomplete_features)]

const NONE<T>: Option<T> = None::<T>;
const IGNORE<T>: () = ();

fn none() {
    let _ = NONE; //~ ERROR type annotations needed
}

fn ignore() {
    let _ = IGNORE; //~ ERROR type annotations needed
}

fn main() {}
