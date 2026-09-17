//! Regression test for <https://github.com/rust-lang/rust/issues/162857>.
//@ check-pass

mod vis1 {
    pub trait Visitor {
        fn visit_b(&self);
        fn visit_c(&self);
    }
}

mod vis2 {
    pub trait Visitor {}
}

use crate::{vis1::*, vis2::*};

pub struct Impl;

impl vis1::Visitor for Impl {
    fn visit_b(&self) {}

    fn visit_c(&self) {
        self.visit_b();
        //~^ WARN: use of ambiguously glob imported trait `Visitor` [ambiguous_glob_imported_traits]
        //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    }
}

fn main() {}
