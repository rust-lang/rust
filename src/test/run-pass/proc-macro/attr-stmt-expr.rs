// aux-build:attr-stmt-expr.rs

#![feature(stmt_expr_attributes, proc_macro_hygiene)]

extern crate attr_stmt_expr;
use attr_stmt_expr::{expect_let, expect_print_stmt, expect_expr, expect_print_expr,
                     no_output, noop};

fn print_str(string: &'static str) {
    // macros are handled a bit differently
    #[expect_print_expr]
    println!("{}", string)
}

fn main() {
    #[expect_let]
    let string = "Hello, world!";

    #[expect_print_stmt]
    println!("{}", string);

    let _: () = {
        #[no_output]
        "Hello, world!"
    };

    let _: &'static str = #[noop] "Hello, world!";

    let _: &'static str = {
        #[noop] "Hello, world!"
    };

    #[expect_expr]
    print_str("string")
}
