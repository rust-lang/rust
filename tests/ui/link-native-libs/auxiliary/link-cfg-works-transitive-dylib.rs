#![feature(link_cfg)]

#[link(name = "foo", cfg(false))]
extern "C" {}
