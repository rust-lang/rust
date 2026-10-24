//@ proc-macro: test-macros.rs
// gate-test-proc_macro_hygiene

#![feature(stmt_expr_attributes)]

#[macro_use]
extern crate test_macros;

fn _test_inner() {
    #![empty_attr] //~ ERROR: inner macro attributes are unstable
}

mod _test2_inner {
    #![empty_attr] //~ ERROR: inner macro attributes are unstable
}

#[empty_attr = "y"] //~ ERROR: key-value macro attributes are not supported
fn _test3() {}

fn attrs() {
    // Statement, item
    #[empty_attr] // OK
    struct S;

    // Statement, macro
    #[empty_attr] //~ ERROR: macro attributes on statements are unstable
    println!();

    // Statement, semi
    #[empty_attr] //~ ERROR: macro attributes on statements are unstable
    S;

    // Statement, local
    #[empty_attr] //~ ERROR: macro attributes on statements are unstable
    let _x = 2;

    // Expr
    let _x = #[identity_attr] 2; //~ ERROR: macro attributes on expressions are unstable

    // Opt expr
    let _x = [#[identity_attr] 2]; //~ ERROR: macro attributes on expressions are unstable

    // Expr macro
    let _x = #[identity_attr] println!();
    //~^ ERROR: macro attributes on expressions are unstable
}

fn test_case() {
    #![test] //~ ERROR inner macro attributes are unstable
}

fn main() {}
