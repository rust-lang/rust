//! Regression test for <https://github.com/rust-lang/rust/issues/22673>.
//@ check-pass

trait Expr: PartialEq<Self::Item> {
    type Item;
}

fn main() {}
