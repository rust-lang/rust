//@ build-fail
//@ compile-flags: --crate-type rlib
//@ error-pattern: could not find native static library `bar.lib`
//@ error-pattern: only provide the library name `bar`, not the full filename

#[link(name = "bar.lib", kind = "static")]
extern { }

pub fn main() { }
