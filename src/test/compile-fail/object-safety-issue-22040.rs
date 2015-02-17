// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #22040.

use std::fmt::Debug;

trait Expr: Debug + PartialEq {
    fn print_element_count(&self);
}

//#[derive(PartialEq)]
#[derive(Debug)]
struct SExpr<'x> {
    elements: Vec<Box<Expr+ 'x>>,
}

impl<'x> PartialEq for SExpr<'x> {
    fn eq(&self, other:&SExpr<'x>) -> bool {
        println!("L1: {} L2: {}", self.elements.len(), other.elements.len());
        let result = self.elements.len() == other.elements.len();

        println!("Got compare {}", result);
        return result;
    }
}

impl <'x> SExpr<'x> {
    fn new() -> SExpr<'x> { return SExpr{elements: Vec::new(),}; }
}

impl <'x> Expr for SExpr<'x> {
    fn print_element_count(&self) {
        println!("element count: {}", self.elements.len());
    }
}

fn main() {
    let a: Box<Expr> = Box::new(SExpr::new()); //~ ERROR trait `Expr` is not object-safe
    let b: Box<Expr> = Box::new(SExpr::new()); //~ ERROR trait `Expr` is not object-safe

    assert_eq!(a , b);
}
