// build-fail
//@compile-flags: --crate-type rlib
//@error-in-other-file: could not find native static library `libfoo.a`
//@error-in-other-file: only provide the library name `foo`, not the full filename

#[link(name = "libfoo.a", kind = "static")]
extern { }

pub fn main() { }
