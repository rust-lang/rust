// compile-flags: -l foo:
// error-pattern: an empty renaming target was specified for library

#![crate_type = "lib"]

#[link(name = "foo")]
extern "C" {}
