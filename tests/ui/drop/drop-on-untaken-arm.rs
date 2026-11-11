//! Regression test for <https://github.com/rust-lang/rust/issues/29092>.
//! Drop glue for match expression `*t.clone()` ran when arm was never taken,
//! causing segfault.
//@ run-pass

use self::Term::*;

#[derive(Clone)]
pub enum Term {
    Dummy,
    A(Box<Term>),
    B(Box<Term>),
}

pub fn small_eval(v: Term) -> Term {
    match v {
        A(t) => *t.clone(),
        B(t) => *t.clone(),
        _ => Dummy,
    }
}

fn main() {
    small_eval(Dummy);
}
