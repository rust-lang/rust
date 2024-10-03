//@ aux-build:unstable_but_const_stable.rs

extern crate unstable_but_const_stable;
use unstable_but_const_stable::*;

fn main() {
    some_unstable_fn(); //~ERROR use of unstable library feature
    unsafe { write_bytes(4 as *mut u8, 0, 0) }; //~ERROR use of unstable library feature
}

const fn const_main() {
    some_unstable_fn(); //~ERROR use of unstable library feature
    unsafe { write_bytes(4 as *mut u8, 0, 0) }; //~ERROR use of unstable library feature
}
