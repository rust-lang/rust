//@ known-bug: #120241
//@ edition:2021
#![feature(object_safe_for_dispatch)]
#![feature(unsized_fn_params)]

fn guard(_s: Copy) -> bool {
    panic!()
}

fn main() {}
