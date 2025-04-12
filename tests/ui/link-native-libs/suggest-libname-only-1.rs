//@ build-fail
//@ compile-flags: --crate-type rlib

#[link(name = "libfoo.a", kind = "static")]
extern { } //~ WARN extern declarations without an explicit ABI are deprecated
           //~| HELP explicitly specify the "C" ABI

pub fn main() { }

//~? ERROR could not find native static library `libfoo.a`
//~? HELP only provide the library name `foo`, not the full filename
