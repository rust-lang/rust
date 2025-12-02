//@ compile-flags: -Zunpretty=thir-tree --crate-type=lib
//@ check-pass
#![feature(c_variadic)]
#![expect(varargs_without_pattern)]

// The `...` argument uses `PatKind::Missing`.
unsafe extern "C" fn foo(_: i32, ...) {}
