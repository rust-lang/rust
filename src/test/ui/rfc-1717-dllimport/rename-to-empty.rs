// compile-flags: -l foo:
// error-pattern: empty override name was specified for library

#![crate_type = "lib"]

#[link(name = "foo")]
extern "C" {}
