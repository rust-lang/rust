//@ check-pass

#![feature(const_debug, const_trait_impl)]

use std::fmt::Debug;
use std::fmt::Formatter;

pub struct Foo;

impl const Debug for Foo {
    fn fmt(&self, _: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        Ok(())
    }
}

fn main() {}
