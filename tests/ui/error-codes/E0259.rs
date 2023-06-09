#![feature(rustc_private)]

extern crate alloc;

extern crate libc as alloc;
//~^ ERROR E0259

fn main() {}
