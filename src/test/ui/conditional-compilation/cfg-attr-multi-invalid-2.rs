//
// compile-flags: --cfg broken

#![feature(cfg_attr_multi)]
#![cfg_attr(broken, no_std, no_core)] //~ ERROR no_core is experimental

fn main() { }
