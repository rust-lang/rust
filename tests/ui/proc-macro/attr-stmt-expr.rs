//@ proc-macro: attr-stmt-expr.rs
//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug

#![feature(proc_macro_hygiene)]
#![feature(rustc_attrs)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;
extern crate test_macros;
extern crate attr_stmt_expr;

use test_macros::print_attr;
use attr_stmt_expr::{expect_let, expect_my_macro_stmt, expect_expr, expect_my_macro_expr};

// We don't use `std::println` so that we avoid loading hygiene
// information from libstd, which would affect the SyntaxContext ids
macro_rules! my_macro {
    ($($tt:tt)*) => { () }
}

fn print_str(string: &'static str) {
    // macros are handled a bit differently
    #[expect_my_macro_expr]
    //~^ ERROR attributes on expressions are experimental
    //~| HELP add `#![feature(stmt_expr_attributes)]` to the crate attributes to enable
    my_macro!("{}", string)
}

macro_rules! make_stmt {
    ($stmt:stmt) => {
        #[print_attr]
        #[rustc_dummy]
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
    #[expect_my_macro_stmt]
    my_macro!("{}", string);

    #[print_attr]
    second_make_stmt!(#[allow(dead_code)] struct Bar {});

    #[print_attr]
    #[rustc_dummy]
    struct Other {}

    #[expect_expr]
    //~^ ERROR attributes on expressions are experimental
    //~| HELP add `#![feature(stmt_expr_attributes)]` to the crate attributes to enable
    print_str("string")
}
