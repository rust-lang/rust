// Here we test that `lowering` behaves correctly wrt. `let $pats = $expr` expressions.
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

#![feature(let_chains)] // Avoid inflating `.stderr` with overzealous gates in this test.

#![allow(irrefutable_let_patterns)]

use std::ops::Range;

fn main() {}

fn _if() {
    if (let 0 = 1) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    if (((let 0 = 1))) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    if (let 0 = 1) && true {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    if true && (let 0 = 1) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    if (let 0 = 1) && (let 0 = 1) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement

    if let 0 = 1 && let 1 = 2 && (let 2 = 3 && let 3 = 4 && let 4 = 5) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
}

fn _while() {
    while (let 0 = 1) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    while (((let 0 = 1))) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    while (let 0 = 1) && true {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    while true && (let 0 = 1) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    while (let 0 = 1) && (let 0 = 1) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement

    while let 0 = 1 && let 1 = 2 && (let 2 = 3 && let 3 = 4 && let 4 = 5) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
}

fn _macros() {
    macro_rules! use_expr {
        ($e:expr) => {
            if $e {}
            while $e {}
        }
    }
    use_expr!((let 0 = 1 && 0 == 0));
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    use_expr!((let 0 = 1));
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
}

fn nested_within_if_expr() {
    if &let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement

    if !let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    if *let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR type `bool` cannot be dereferenced
    //~| ERROR expected expression, found `let` statement
    if -let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR cannot apply unary operator `-` to type `bool`
    //~| ERROR expected expression, found `let` statement

    fn _check_try_binds_tighter() -> Result<(), ()> {
        if let 0 = 0? {}
        //~^ ERROR the `?` operator can only be applied to values that implement `Try`
        Ok(())
    }
    if (let 0 = 0)? {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR the `?` operator can only be applied to values that implement `Try`
    //~| ERROR the `?` operator can only be used in a function that returns `Result`
    //~| ERROR expected expression, found `let` statement

    if true || let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    if (true || let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    if true && (true || let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    if true || (true && let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    let mut x = true;
    if x = let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement

    if true..(let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
    if ..(let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
    if (let 0 = 0).. {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement

    // Binds as `(let ... = true)..true &&/|| false`.
    if let Range { start: _, end: _ } = true..true && false {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types
    if let Range { start: _, end: _ } = true..true || false {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    // Binds as `(let Range { start: F, end } = F)..(|| true)`.
    const F: fn() -> bool = || true;
    if let Range { start: F, end } = F..|| true {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    // Binds as `(let Range { start: true, end } = t)..(&&false)`.
    let t = &&true;
    if let Range { start: true, end } = t..&&false {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    if let true = let true = true {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
}

fn nested_within_while_expr() {
    while &let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement

    while !let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    while *let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR type `bool` cannot be dereferenced
    //~| ERROR expected expression, found `let` statement
    while -let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR cannot apply unary operator `-` to type `bool`
    //~| ERROR expected expression, found `let` statement

    fn _check_try_binds_tighter() -> Result<(), ()> {
        while let 0 = 0? {}
        //~^ ERROR the `?` operator can only be applied to values that implement `Try`
        Ok(())
    }
    while (let 0 = 0)? {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR the `?` operator can only be applied to values that implement `Try`
    //~| ERROR the `?` operator can only be used in a function that returns `Result`
    //~| ERROR expected expression, found `let` statement

    while true || let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    while (true || let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    while true && (true || let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    while true || (true && let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    let mut x = true;
    while x = let 0 = 0 {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement

    while true..(let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
    while ..(let 0 = 0) {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
    while (let 0 = 0).. {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement

    // Binds as `(let ... = true)..true &&/|| false`.
    while let Range { start: _, end: _ } = true..true && false {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types
    while let Range { start: _, end: _ } = true..true || false {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    // Binds as `(let Range { start: F, end } = F)..(|| true)`.
    const F: fn() -> bool = || true;
    while let Range { start: F, end } = F..|| true {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    // Binds as `(let Range { start: true, end } = t)..(&&false)`.
    let t = &&true;
    while let Range { start: true, end } = t..&&false {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    while let true = let true = true {}
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
}

fn not_error_because_clarified_intent() {
    if let Range { start: _, end: _ } = (true..true || false) { }

    if let Range { start: _, end: _ } = (true..true && false) { }

    while let Range { start: _, end: _ } = (true..true || false) { }

    while let Range { start: _, end: _ } = (true..true && false) { }
}

fn outside_if_and_while_expr() {
    &let 0 = 0;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    !let 0 = 0;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    *let 0 = 0;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR type `bool` cannot be dereferenced
    //~| ERROR expected expression, found `let` statement
    -let 0 = 0;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR cannot apply unary operator `-` to type `bool`
    //~| ERROR expected expression, found `let` statement

    fn _check_try_binds_tighter() -> Result<(), ()> {
        let 0 = 0?;
        //~^ ERROR the `?` operator can only be applied to values that implement `Try`
        Ok(())
    }
    (let 0 = 0)?;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR the `?` operator can only be used in a function that returns `Result`
    //~| ERROR the `?` operator can only be applied to values that implement `Try`
    //~| ERROR expected expression, found `let` statement

    true || let 0 = 0;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    (true || let 0 = 0);
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    true && (true || let 0 = 0);
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    let mut x = true;
    x = let 0 = 0;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    true..(let 0 = 0);
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    ..(let 0 = 0);
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    (let 0 = 0)..;
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement

    (let Range { start: _, end: _ } = true..true || false);
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement

    (let true = let true = true);
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement

    {
        #[cfg(FALSE)]
        let x = true && let y = 1;
        //~^ ERROR expected expression, found `let` statement
    }

    #[cfg(FALSE)]
    {
        [1, 2, 3][let _ = ()]
        //~^ ERROR expected expression, found `let` statement
    }

    // Check function tail position.
    &let 0 = 0
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
}

// Let's make sure that `let` inside const generic arguments are considered.
fn inside_const_generic_arguments() {
    struct A<const B: bool>;
    impl<const B: bool> A<{B}> { const O: u32 = 5; }

    if let A::<{
        true && let 1 = 1
        //~^ ERROR `let` expressions are not supported here
        //~| ERROR expected expression, found `let` statement
    }>::O = 5 {}

    while let A::<{
        true && let 1 = 1
        //~^ ERROR `let` expressions are not supported here
        //~| ERROR expected expression, found `let` statement
    }>::O = 5 {}

    if A::<{
        true && let 1 = 1
        //~^ ERROR `let` expressions are not supported here
        //~| ERROR expected expression, found `let` statement
    }>::O == 5 {}

    // In the cases above we have `ExprKind::Block` to help us out.
    // Below however, we would not have a block and so an implementation might go
    // from visiting expressions to types without banning `let` expressions down the tree.
    // This tests ensures that we are not caught by surprise should the parser
    // admit non-IDENT expressions in const generic arguments.

    if A::<
        true && let 1 = 1
        //~^ ERROR `let` expressions are not supported here
        //~| ERROR expressions must be enclosed in braces
        //~| ERROR expected expression, found `let` statement
    >::O == 5 {}
}

fn with_parenthesis() {
    let opt = Some(Some(1i32));

    if (let Some(a) = opt && true) {
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    }

    if (let Some(a) = opt) && true {
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    }
    if (let Some(a) = opt) && (let Some(b) = a) {
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
    }
    if let Some(a) = opt && (true && true) {
    }

    if (let Some(a) = opt && (let Some(b) = a)) && b == 1 {
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
    }
    if (let Some(a) = opt && (let Some(b) = a)) && true {
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
    }
    if (let Some(a) = opt && (true)) && true {
    //~^ ERROR `let` expressions are not supported here
    //~| ERROR expected expression, found `let` statement
    }

    if (true && (true)) && let Some(a) = opt {
    }
    if (true) && let Some(a) = opt {
    }
    if true && let Some(a) = opt {
    }

    let fun = || true;
    if let true = (true && fun()) && (true) {
    }

    #[cfg(FALSE)]
    let x = (true && let y = 1);
    //~^ ERROR expected expression, found `let` statement

    #[cfg(FALSE)]
    {
        ([1, 2, 3][let _ = ()])
        //~^ ERROR expected expression, found `let` statement
    }
}
