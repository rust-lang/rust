//@ build-fail
//@ compile-flags: --crate-type rlib
//@ error-pattern: only provide the library name `bar`, not the full filename

#[link(name = "bar.lib", kind = "static")]
extern { } //~ WARN extern declarations without an explicit ABI are deprecated

pub fn main() { }

//~? ERROR could not find native static library `bar.lib`
