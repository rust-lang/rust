//@ aux-build:internal_unstable.rs

#![allow(dead_code)]

extern crate internal_unstable;


thread_local!(static FOO: () = ());
thread_local!(static BAR: () = internal_unstable::unstable()); //~ ERROR use of unstable

fn main() {}
