#![feature(link_cfg)]
#![crate_type = "rlib"]

#[link(name = "testb", cfg(foo))]
extern "C" {}

#[link(name = "testb", cfg(bar))]
extern "C" {}
