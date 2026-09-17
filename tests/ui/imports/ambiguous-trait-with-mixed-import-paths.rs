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
    }
}

fn main() {}
