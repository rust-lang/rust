//@ build-fail
//@ compile-flags: --crate-type rlib

#[link(name = "bar.lib", kind = "static")]
extern { } //~ WARN `extern` declarations without an explicit ABI are deprecated
           //~| HELP explicitly specify the "C" ABI

pub fn main() { }

//~? ERROR could not find native static library `bar.lib`
//~? HELP only provide the library name `bar`, not the full filename
