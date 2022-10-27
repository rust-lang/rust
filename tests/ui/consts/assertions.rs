#![feature(const_assert_eq)]

const _BAD1: () = {
    assert_eq!(1, 2)
}; //~^ ERROR: evaluation of constant value failed

const _BAD2: () = {
    assert_ne!(1, 1)
}; //~^ ERROR: evaluation of constant value failed

const _BAD3: () = {
    assert!(false)
}; //~^ ERROR: evaluation of constant value failed

fn main() {}
