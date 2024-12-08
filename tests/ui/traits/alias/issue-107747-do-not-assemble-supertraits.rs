// Regression test for #107747: methods from trait alias supertraits were brought into scope
//
//@ check-pass

#![feature(trait_alias)]

use std::fmt;

trait Foo: fmt::Debug {}
trait Bar = Foo;

#[derive(Debug)]
struct Qux(bool);

impl fmt::Display for Qux {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

fn main() {}
