// aux-build:proc-macro-gates.rs

#![feature(stmt_expr_attributes)]

extern crate proc_macro_gates as foo;

use foo::*;

// NB. these errors aren't the best errors right now, but they're definitely
// intended to be errors. Somehow using a custom attribute in these positions
// should either require a feature gate or not be allowed on stable.

fn _test6<#[a] T>() {}
//~^ ERROR: unknown to the compiler

fn _test7() {
    match 1 {
        #[a] //~ ERROR: unknown to the compiler
        0 => {}
        _ => {}
    }
}

fn main() {
}
