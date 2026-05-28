//@ check-pass
//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/221>.
// Ensure that we normalize after applying projection elems in MIR typeck.

use std::marker::PhantomData;

#[derive(Copy, Clone)]
struct Span;

trait AstKind {
    type Inner;
}

struct WithSpan;
impl AstKind for WithSpan {
    type Inner
        = (i32,);
}

struct Expr<'a> { f: &'a <WithSpan as AstKind>::Inner }

impl Expr<'_> {
    fn span(self) {
        match self {
            Self { f: (n,) } => {},
        }
    }
}

fn main() {}
