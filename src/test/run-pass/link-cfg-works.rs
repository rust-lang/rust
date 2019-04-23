// aux-build:link-cfg-works-transitive-rlib.rs
// aux-build:link-cfg-works-transitive-dylib.rs

#![feature(link_cfg)]

extern crate link_cfg_works_transitive_rlib;
extern crate link_cfg_works_transitive_dylib;

#[link(name = "foo", cfg(foo))]
extern {}

fn main() {}
