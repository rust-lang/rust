//@ compile-flags: -Zunpretty=thir-tree --crate-type=lib
//@ check-pass

unsafe extern "C" fn foo(_: i32, _: ...) {}
