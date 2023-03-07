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
use attr_stmt_expr::{expect_let, expect_my_macro_stmt, expect_expr, expect_my_macro_expr};
use test_macros::print_attr;

// We don't use `std::println` so that we avoid loading hygiene
// information from libstd, which would affect the SyntaxContext ids
macro_rules! my_macro {
    ($($tt:tt)*) => { () }
}


fn print_str(string: &'static str) {
    // macros are handled a bit differently
    #[expect_my_macro_expr]
    my_macro!("{}", string)
}

macro_rules! make_stmt {
    ($stmt:stmt) => {
        #[print_attr]
        #[rustc_dummy]
        $stmt; // This semicolon is *not* passed to the macro,
               // since `$stmt` is already a statement.
    }
}

macro_rules! second_make_stmt {
    ($stmt:stmt) => {
        make_stmt!($stmt);
    }
}

// The macro will see a semicolon here
#[print_attr]
struct ItemWithSemi;


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
    struct Other {};

    // The macro also sees a semicolon,
    // for consistency with the `ItemWithSemi` case above.
    #[print_attr]
    #[rustc_dummy]
    struct NonBracedStruct;

    #[expect_expr]
    print_str("string")
}
