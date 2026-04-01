//@ edition:2021
// Test that it doesn't trigger an ICE when using an unsized fn params.
// https://github.com/rust-lang/rust/issues/120241

#![feature(unsized_fn_params)]

fn guard(_s: Copy) -> bool {
    //~^ ERROR: expected a type, found a trait
    panic!()
}

fn main() {}
