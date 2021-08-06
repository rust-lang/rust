// Test native_link_modifiers_bundle don't need static-nobundle
// check-pass

#![feature(native_link_modifiers)]
#![feature(native_link_modifiers_bundle)]

#[link(name = "foo", kind = "static", modifiers = "-bundle")]
extern "C" {}

fn main() {}
