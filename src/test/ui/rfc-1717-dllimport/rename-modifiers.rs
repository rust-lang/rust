// compile-flags: -l dylib=foo:bar
// error-pattern: override for library `foo` must specify modifiers

#![feature(native_link_modifiers_as_needed)]

#![crate_type = "lib"]

#[link(name = "foo", kind = "dylib", modifiers = "-as-needed")]
extern "C" {}
