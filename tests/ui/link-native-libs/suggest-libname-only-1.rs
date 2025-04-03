//@ build-fail
//@ compile-flags: --crate-type rlib
//@ error-pattern: only provide the library name `foo`, not the full filename

#[link(name = "libfoo.a", kind = "static")]
extern { } //~ WARN extern declarations without an explicit ABI are deprecated

pub fn main() { }

//~? ERROR could not find native static library `libfoo.a`
