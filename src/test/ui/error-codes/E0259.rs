#![feature(rustc_private)]
#![allow(unused_extern_crates)]

extern crate alloc;

extern crate libc as alloc;
//~^ ERROR E0259

fn main() {}
