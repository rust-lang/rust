// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
