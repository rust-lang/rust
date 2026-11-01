// A regression test for issue #81099.

//@ check-pass
//@ proc-macro:test-macros.rs

#![feature(stmt_expr_attributes)]
#![feature(proc_macro_hygiene)]

#[macro_use]
extern crate test_macros;

#[derive(Clone, Copy)]
struct S {
    // `print_args` runs twice
    // - on eagerly configured `S` (from `impl Copy`), only 11 should be printed
    // - on non-configured `S` (from `struct S`), both 10 and 11 should be printed
    field: [u8; #[print_attr] {
        #[cfg(false)] { 10 }
        #[cfg(not(FALSE))]  { 11 }
    }],
}

fn main() {}
