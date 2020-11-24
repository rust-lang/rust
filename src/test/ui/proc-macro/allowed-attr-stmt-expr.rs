// aux-build:attr-stmt-expr.rs
// aux-build:test-macros.rs
// compile-flags: -Z span-debug
// check-pass

#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate attr_stmt_expr;
extern crate test_macros;
use attr_stmt_expr::{expect_let, expect_print_stmt, expect_expr, expect_print_expr};
use test_macros::print_attr;
use std::println;

fn print_str(string: &'static str) {
    // macros are handled a bit differently
    #[expect_print_expr]
    println!("{}", string)
}

macro_rules! make_stmt {
    ($stmt:stmt) => {
        $stmt
    }
}

macro_rules! second_make_stmt {
    ($stmt:stmt) => {
        make_stmt!($stmt);
    }
}


fn main() {
    make_stmt!(struct Foo {});

    #[print_attr]
    #[expect_let]
    let string = "Hello, world!";

    #[print_attr]
    #[expect_print_stmt]
    println!("{}", string);

    #[print_attr]
    second_make_stmt!(#[allow(dead_code)] struct Bar {});

    #[print_attr]
    #[rustc_dummy]
    struct Other {};

    #[expect_expr]
    print_str("string")
}
