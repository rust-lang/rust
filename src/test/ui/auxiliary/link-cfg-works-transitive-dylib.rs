#![feature(link_cfg)]

#[link(name = "foo", cfg(foo))]
extern {}
