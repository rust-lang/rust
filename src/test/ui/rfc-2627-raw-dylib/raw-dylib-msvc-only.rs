// only-windows-gnu
// check-pass
// compile-flags: --crate-type lib
#![feature(raw_dylib)]
//~^ WARNING: the feature `raw_dylib` is incomplete
#[link(name = "foo", kind = "raw-dylib")]
//~^ WARNING: `#[link(...)]` with `kind = "raw-dylib"` not supported on windows-gnu
extern "C" {}
