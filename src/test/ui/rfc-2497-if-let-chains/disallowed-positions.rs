// Here we test that `ast_validation` behaves correctly wrt. `let $pats = $expr` expressions.
//
// We want to make sure that `let` is banned in situations other than:
//
// expr =
//   | ...
//   | "if" expr_with_let block {"else" block}?
//   | {label ":"}? while" expr_with_let block
//   ;
//
// expr_with_let =
//   | "let" top_pats "=" expr
//   | expr_with_let "&&" expr_with_let
//   | "(" expr_with_let ")"
//   | expr
//   ;
//
// To that end, we check some positions which is not part of the language above.

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete
#![feature(let_chains)] // Avoid inflating `.stderr` with overzealous gates in this test.
//~^ WARN the feature `let_chains` is incomplete

#![allow(irrefutable_let_patterns)]

use std::ops::Range;

fn main() {}

fn nested_within_if_expr() {
    if &let 0 = 0 {} //~ ERROR `let` expressions are not supported here
    //~^ ERROR `let` expressions only supported in `if`

    if !let 0 = 0 {} //~ ERROR `let` expressions are not supported here
    if *let 0 = 0 {} //~ ERROR `let` expressions are not supported here
    if -let 0 = 0 {} //~ ERROR `let` expressions are not supported here

    fn _check_try_binds_tighter() -> Result<(), ()> {
        if let 0 = 0? {}
        Ok(())
    }
    if (let 0 = 0)? {} //~ ERROR `let` expressions are not supported here

    if true || let 0 = 0 {} //~ ERROR `let` expressions are not supported here
    if (true || let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    if true && (true || let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    if true || (true && let 0 = 0) {} //~ ERROR `let` expressions are not supported here

    let mut x = true;
    if x = let 0 = 0 {} //~ ERROR `let` expressions are not supported here

    if true..(let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    if ..(let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    if (let 0 = 0).. {} //~ ERROR `let` expressions are not supported here

    // Binds as `(let ... = true)..true &&/|| false`.
    if let Range { start: _, end: _ } = true..true && false {}
    //~^ ERROR `let` expressions are not supported here
    if let Range { start: _, end: _ } = true..true || false {}
    //~^ ERROR `let` expressions are not supported here

    // Binds as `(let Range { start: F, end } = F)..(|| true)`.
    const F: fn() -> bool = || true;
    if let Range { start: F, end } = F..|| true {}
    //~^ ERROR `let` expressions are not supported here

    // Binds as `(let Range { start: true, end } = t)..(&&false)`.
    let t = &&true;
    if let Range { start: true, end } = t..&&false {}
    //~^ ERROR `let` expressions are not supported here

    if let true = let true = true {} //~ ERROR `let` expressions are not supported here
}

fn nested_within_while_expr() {
    while &let 0 = 0 {} //~ ERROR `let` expressions are not supported here

    while !let 0 = 0 {} //~ ERROR `let` expressions are not supported here
    while *let 0 = 0 {} //~ ERROR `let` expressions are not supported here
    while -let 0 = 0 {} //~ ERROR `let` expressions are not supported here

    fn _check_try_binds_tighter() -> Result<(), ()> {
        while let 0 = 0? {}
        Ok(())
    }
    while (let 0 = 0)? {} //~ ERROR `let` expressions are not supported here

    while true || let 0 = 0 {} //~ ERROR `let` expressions are not supported here
    while (true || let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    while true && (true || let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    while true || (true && let 0 = 0) {} //~ ERROR `let` expressions are not supported here

    let mut x = true;
    while x = let 0 = 0 {} //~ ERROR `let` expressions are not supported here

    while true..(let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    while ..(let 0 = 0) {} //~ ERROR `let` expressions are not supported here
    while (let 0 = 0).. {} //~ ERROR `let` expressions are not supported here

    // Binds as `(let ... = true)..true &&/|| false`.
    while let Range { start: _, end: _ } = true..true && false {}
    //~^ ERROR `let` expressions are not supported here
    while let Range { start: _, end: _ } = true..true || false {}
    //~^ ERROR `let` expressions are not supported here

    // Binds as `(let Range { start: F, end } = F)..(|| true)`.
    const F: fn() -> bool = || true;
    while let Range { start: F, end } = F..|| true {}
    //~^ ERROR `let` expressions are not supported here

    // Binds as `(let Range { start: true, end } = t)..(&&false)`.
    let t = &&true;
    while let Range { start: true, end } = t..&&false {}
    //~^ ERROR `let` expressions are not supported here

    while let true = let true = true {} //~ ERROR `let` expressions are not supported here
}

fn not_error_because_clarified_intent() {
    if let Range { start: _, end: _ } = (true..true || false) { }

    if let Range { start: _, end: _ } = (true..true && false) { }

    while let Range { start: _, end: _ } = (true..true || false) { }

    while let Range { start: _, end: _ } = (true..true && false) { }
}

fn outside_if_and_while_expr() {
    &let 0 = 0; //~ ERROR `let` expressions are not supported here

    !let 0 = 0; //~ ERROR `let` expressions are not supported here
    *let 0 = 0; //~ ERROR `let` expressions are not supported here
    -let 0 = 0; //~ ERROR `let` expressions are not supported here

    fn _check_try_binds_tighter() -> Result<(), ()> {
        let 0 = 0?;
        Ok(())
    }
    (let 0 = 0)?; //~ ERROR `let` expressions are not supported here

    true || let 0 = 0; //~ ERROR `let` expressions are not supported here
    (true || let 0 = 0); //~ ERROR `let` expressions are not supported here
    true && (true || let 0 = 0); //~ ERROR `let` expressions are not supported here

    let mut x = true;
    x = let 0 = 0; //~ ERROR `let` expressions are not supported here

    true..(let 0 = 0); //~ ERROR `let` expressions are not supported here
    ..(let 0 = 0); //~ ERROR `let` expressions are not supported here
    (let 0 = 0)..; //~ ERROR `let` expressions are not supported here

    (let Range { start: _, end: _ } = true..true || false);
    //~^ ERROR `let` expressions are not supported here

    (let true = let true = true);
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here

    // Check function tail position.
    &let 0 = 0
    //~^ ERROR `let` expressions are not supported here
}

// Let's make sure that `let` inside const generic arguments are considered.
fn inside_const_generic_arguments() {
    struct A<const B: bool>;
    impl<const B: bool> A<{B}> { const O: u32 = 5; }

    if let A::<{
        true && let 1 = 1 //~ ERROR `let` expressions are not supported here
    }>::O = 5 {}

    while let A::<{
        true && let 1 = 1 //~ ERROR `let` expressions are not supported here
    }>::O = 5 {}

    if A::<{
        true && let 1 = 1 //~ ERROR `let` expressions are not supported here
    }>::O == 5 {}

    // In the cases above we have `ExprKind::Block` to help us out.
    // Below however, we would not have a block and so an implementation might go
    // from visiting expressions to types without banning `let` expressions down the tree.
    // This tests ensures that we are not caught by surprise should the parser
    // admit non-IDENT expressions in const generic arguments.

    if A::<
        true && let 1 = 1 //~ ERROR expected one of `,` or `>`, found `&&`
    >::O == 5 {}
}
