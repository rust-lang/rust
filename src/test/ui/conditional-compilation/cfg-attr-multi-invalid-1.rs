//
// compile-flags: --cfg broken

#![feature(cfg_attr_multi)]
#![cfg_attr(broken, no_core, no_std)] //~ ERROR no_core is experimental

fn main() { }
