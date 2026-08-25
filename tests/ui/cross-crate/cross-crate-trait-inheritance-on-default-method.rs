//@ run-pass
#![allow(dead_code)]
//@ aux-build:cross-crate-trait-inheritance-on-default-method.rs
//! Regression test for https://github.com/rust-lang/rust/issues/3979
//! Exposed a bug where external traits on a default method were failing to typecheck.

extern crate cross_crate_trait_inheritance_on_default_method as  issue_3979_traits;
use issue_3979_traits::{Positioned, Movable};

struct Point { x: isize, y: isize }

impl Positioned for Point {
    fn SetX(&mut self, x: isize) {
        self.x = x;
    }
    fn X(&self) -> isize {
        self.x
    }
}

impl Movable for Point {}

pub fn main() {
    let mut p = Point{ x: 1, y: 2};
    p.translate(3);
    assert_eq!(p.X(), 4);
}
