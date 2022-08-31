// only-windows
// only-x86
#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete

#[link(name = "foo", kind = "raw-dylib", import_name_type = "unknown")]
//~^ ERROR unknown import name type `unknown`, expected one of: decorated, noprefix, undecorated
extern "C" { }

fn main() {}
