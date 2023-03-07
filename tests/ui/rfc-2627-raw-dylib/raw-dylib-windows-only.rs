// ignore-windows
// compile-flags: --crate-type lib
#![cfg_attr(target_arch = "x86", feature(raw_dylib))]
#[link(name = "foo", kind = "raw-dylib")]
//~^ ERROR: link kind `raw-dylib` is only supported on Windows targets
extern "C" {}
