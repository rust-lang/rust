#![deny(deprecated)]
#![feature(no_debug)]

#[no_debug] //~ ERROR use of deprecated attribute `no_debug`
fn main() {}
