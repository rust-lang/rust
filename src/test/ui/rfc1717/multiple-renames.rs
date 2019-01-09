// compile-flags: -l foo:bar -l foo:baz
// error-pattern: multiple renamings were specified for library

#![crate_type = "lib"]

#[link(name = "foo")]
extern "C" {}
