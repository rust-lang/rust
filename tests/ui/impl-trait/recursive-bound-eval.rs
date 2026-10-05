//! Test that we can evaluate nested obligations when invoking methods on recursive calls on
//! an RPIT.

//@ revisions: next current
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

pub trait Parser<E> {
    fn parse(&self) -> E;
}

impl<E, T: Fn() -> E> Parser<E> for T {
    fn parse(&self) -> E {
        self()
    }
}

pub fn recursive_fn<E>() -> impl Parser<E> {
    move || recursive_fn().parse()
}

fn main() {}
