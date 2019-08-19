// aux-build:attr-stmt-expr.rs

//! Attributes producing expressions in invalid locations

#![feature(stmt_expr_attributes, proc_macro_hygiene)]

extern crate attr_stmt_expr;
use attr_stmt_expr::{duplicate, no_output};

fn main() {
    let _ = #[no_output] "Hello, world!";
    //~^ ERROR expected expression, found end of macro arguments

    let _ = #[duplicate] "Hello, world!";
    //~^ ERROR macro expansion ignores token `,` and any following

    let _ = {
        #[no_output]
        "Hello, world!"
    };

    let _ = {
        #[duplicate]
        //~^ ERROR macro expansion ignores token `,` and any following
        "Hello, world!"
    };
}
