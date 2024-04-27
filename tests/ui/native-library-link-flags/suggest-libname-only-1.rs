//@ build-fail
//@ compile-flags: --crate-type rlib
//@ error-pattern: could not find native static library `libfoo.a`
//@ error-pattern: only provide the library name `foo`, not the full filename

#[link(name = "libfoo.a", kind = "static")]
extern { }

pub fn main() { }
