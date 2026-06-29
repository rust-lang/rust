//! Regression test for <https://github.com/rust-lang/rust/issues/25497>.
//! Test box deref in match arm doesn't generate invalid LLVM IR.
//! Related <https://github.com/rust-lang/rust/issues/18845>.
//@ run-pass

#[derive(Clone, Debug, PartialEq)]
enum Expression {
    Dummy,
    Add(Box<Expression>),
}

use Expression::*;

fn simplify(exp: Expression) -> Expression {
    match exp {
        Add(n) => *n.clone(),
        _ => Dummy
    }
}

fn main() {
    assert_eq!(simplify(Add(Box::new(Dummy))), Dummy);
}
