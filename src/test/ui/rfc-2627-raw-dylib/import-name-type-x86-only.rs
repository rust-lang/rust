// only-windows
// ignore-x86
#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete
#[link(name = "foo", kind = "raw-dylib", import_name_type = "decorated")]
//~^ ERROR import name type is only supported on x86
extern "C" { }

fn main() {}
