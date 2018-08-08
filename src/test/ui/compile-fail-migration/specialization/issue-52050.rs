// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

// Regression test for #52050: when inserting the blanket impl `I`
// into the tree, we had to replace the child node for `Foo`, which
// led to the struture of the tree being messed up.

use std::iter::Iterator;

trait IntoPyDictPointer { }

struct Foo { }

impl Iterator for Foo {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        None
    }
}

impl IntoPyDictPointer for Foo { }

impl<I> IntoPyDictPointer for I
where
    I: Iterator,
{
}

impl IntoPyDictPointer for () //~ ERROR conflicting implementations
{
}

fn main() { }
