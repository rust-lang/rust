//@ compile-flags: -Zunpretty=thir-tree --crate-type=lib
//@ check-pass
#![feature(c_variadic)]

// The `...` argument uses `PatKind::Missing`.
unsafe extern "C" fn foo(_: i32, ...) {}
