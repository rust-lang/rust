// compile-flags: -Zunstable-options --crate-type rlib
// build-fail
// error-pattern: the linking modifiers `+bundle` and `+whole-archive` are not compatible with each other when generating rlibs

#![feature(native_link_modifiers)]
#![feature(native_link_modifiers_bundle)]
#![feature(native_link_modifiers_whole_archive)]

#[link(name = "mylib", kind = "static", modifiers = "+bundle,+whole-archive")]
extern "C" { }

fn main() { }
