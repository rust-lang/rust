//@ known-bug: #120241
//@ edition:2021
#![feature(dyn_compatible_for_dispatch)]
#![feature(unsized_fn_params)]

fn guard(_s: Copy) -> bool {
    panic!()
}

fn main() {}
