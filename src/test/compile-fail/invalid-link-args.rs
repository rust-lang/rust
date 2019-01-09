// ignore-msvc due to linker-flavor=ld
// error-pattern:aFdEfSeVEEE
// compile-flags: -C linker-flavor=ld

/* Make sure invalid link_args are printed to stderr. */

#![feature(link_args)]

#[link_args = "aFdEfSeVEEE"]
extern {}

fn main() { }
