// compile-flags: -Zunstable-options --crate-type rlib
// build-fail
// error-pattern: the linking modifiers `+bundle` and `+whole-archive` are not compatible with each other when generating rlibs

#![feature(native_link_modifiers_bundle)]

#[link(name = "mylib", kind = "static", modifiers = "+bundle,+whole-archive")]
extern "C" { }

fn main() { }
