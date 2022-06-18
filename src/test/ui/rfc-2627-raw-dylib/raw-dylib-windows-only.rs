// ignore-windows
// compile-flags: --crate-type lib
#![feature(raw_dylib)]
//~^ WARNING: the feature `raw_dylib` is incomplete
#[link(name = "foo", kind = "raw-dylib")]
//~^ ERROR: link kind `raw-dylib` is only supported on Windows targets
extern "C" {}
