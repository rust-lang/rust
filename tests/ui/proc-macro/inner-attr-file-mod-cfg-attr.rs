//@ check-pass
//@ proc-macro: test-macros.rs
//@ edition:2024
//@ compile-flags: --cfg enabled --check-cfg=cfg(enabled)

#![feature(custom_inner_attributes)]
#![deny(dead_code, unused_attributes, unused_imports)]

extern crate test_macros;

#[path = "auxiliary/inner_attr_file_mod_cfg_attr_child.rs"]
mod child;

fn main() {}
