#![feature(packed_bundled_libs)]

#[link(name = "native_dep_1", kind = "static", modifiers = "+whole-archive,+bundle")]
extern "C" {}

#[link(name = "native_dep_2", kind = "static", modifiers = "+whole-archive,-bundle")]
extern "C" {}

#[link(name = "native_dep_3", kind = "static", modifiers = "+bundle")]
extern "C" {}

#[link(name = "native_dep_4", kind = "static", modifiers = "-bundle")]
extern "C" {}

#[no_mangle]
pub fn rust_dep() {}
