// build-fail
//@compile-flags: --crate-type rlib
//@error-in-other-file: could not find native static library `bar.lib`
//@error-in-other-file: only provide the library name `bar`, not the full filename

#[link(name = "bar.lib", kind = "static")]
extern { }

pub fn main() { }
