//@ run-pass
// Regression test for Issue #29092.
//
// (Possibly redundant with regression test run-pass/issue-30530.rs)

use self::Term::*;

#[derive(Clone)]
pub enum Term {
    Dummy,
    A(Box<Term>),
    B(Box<Term>),
}

// a small-step evaluator
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
