// aux-build:proc-macro-gates.rs
// gate-test-proc_macro_hygiene

#![feature(stmt_expr_attributes)]

extern crate proc_macro_gates as foo;

use foo::*;

fn _test_inner() {
    #![a] //~ ERROR: non-builtin inner attributes are unstable
}

#[a] //~ ERROR: custom attributes cannot be applied to modules
mod _test2 {}

mod _test2_inner {
    #![a] //~ ERROR: custom attributes cannot be applied to modules
          //~| ERROR: non-builtin inner attributes are unstable
}

#[a = "y"] //~ ERROR: must only be followed by a delimiter token
fn _test3() {}

fn attrs() {
    // Statement, item
    #[a] // OK
    struct S;

    // Statement, macro
    #[a] //~ ERROR: custom attributes cannot be applied to statements
    println!();

    // Statement, semi
    #[a] //~ ERROR: custom attributes cannot be applied to statements
    S;

    // Statement, local
    #[a] //~ ERROR: custom attributes cannot be applied to statements
    let _x = 2;

    // Expr
    let _x = #[a] 2; //~ ERROR: custom attributes cannot be applied to expressions

    // Opt expr
    let _x = [#[a] 2]; //~ ERROR: custom attributes cannot be applied to expressions

    // Expr macro
    let _x = #[a] println!(); //~ ERROR: custom attributes cannot be applied to expressions
}

fn main() {
    let _x: m!(u32) = 3; //~ ERROR: procedural macros cannot be expanded to types
    if let m!(Some(_x)) = Some(3) {} //~ ERROR: procedural macros cannot be expanded to patterns

    m!(struct S;); //~ ERROR: procedural macros cannot be expanded to statements
    m!(let _x = 3;); //~ ERROR: procedural macros cannot be expanded to statements

    let _x = m!(3); //~ ERROR: procedural macros cannot be expanded to expressions
    let _x = [m!(3)]; //~ ERROR: procedural macros cannot be expanded to expressions
}
