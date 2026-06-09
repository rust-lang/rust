//@ check-pass

#![feature(unsized_fn_params)]

pub fn f0(_f: dyn FnOnce()) {}

fn main() {}
