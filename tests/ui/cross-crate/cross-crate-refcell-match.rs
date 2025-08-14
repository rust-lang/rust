//! Regression test for https://github.com/rust-lang/rust/issues/16822
//
//! ICE when using RefCell::borrow_mut()
//! inside match statement with cross-crate generics.
//!
//! The bug occurred when:
//! - A library defines a generic struct with RefCell<T> and uses borrow_mut() in match
//! - Main crate implements the library trait for its own type
//! - Cross-crate generic constraint causes type inference issues
//!
//! The problematic match statement is in the auxiliary file, this file triggers it.

//@ run-pass
//@ aux-build:cross-crate-refcell-match.rs

extern crate cross_crate_refcell_match as lib;

use std::cell::RefCell;

struct App {
    i: isize,
}

impl lib::Update for App {
    fn update(&mut self) {
        self.i += 1;
    }
}

fn main() {
    let app = App { i: 5 };
    let window = lib::Window { data: RefCell::new(app) };
    // This specific pattern (RefCell::borrow_mut in match with cross-crate generics)
    // caused the ICE in the original issue
    window.update(1);
}
