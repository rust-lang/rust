//@ no-prefer-dynamic

#![feature(link_cfg)]
#![crate_type = "rlib"]

#[link(name = "foo", cfg(false))]
extern "C" {}
