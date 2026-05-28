#![feature(link_cfg)]

#[link(name = "native_dep_1", kind = "static", cfg(should_add))]
extern "C" {}

#[link(name = "native_dep_2", kind = "static", cfg(should_not_add))]
extern "C" {}

#[no_mangle]
pub fn rust_dep() {}
