//@ compile-flags: -Z unstable-options --generate-redirect-map

#![crate_name = "foo"]

//@ !has foo/redirect-map.json
pub struct Foo;
