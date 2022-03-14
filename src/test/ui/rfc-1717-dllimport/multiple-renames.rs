// compile-flags: -l foo:bar -l foo:baz
// error-pattern: multiple overrides were specified for library

#![crate_type = "lib"]

#[link(name = "foo")]
extern "C" {}
