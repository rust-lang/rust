//! Test that `super let` bindings in `if` expressions' blocks have the same scope as the result
//! of the block.
//@ check-pass
#![feature(super_let)]

fn main() {
    // For `super let` in an extending `if`, the binding `temp` should live in the scope of the
    // outer `let` statement.
    let x = if true {
        super let temp = ();
        &temp
    } else {
        super let temp = ();
        &temp
    };
    x;

    // For `super let` in non-extending `if`, the binding `temp` should live in the temporary scope
    // the `if` expression is in.
    std::convert::identity(if true {
        super let temp = ();
        &temp
    } else {
        super let temp = ();
        &temp
    });
}
