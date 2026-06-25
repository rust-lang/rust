//@ compile-flags: -Zunpretty=thir-tree --crate-type=lib
//@ check-pass
#![expect(varargs_without_pattern)]

// The `...` argument uses `PatKind::Missing`.
unsafe extern "C" fn foo(_: i32, ...) {}
