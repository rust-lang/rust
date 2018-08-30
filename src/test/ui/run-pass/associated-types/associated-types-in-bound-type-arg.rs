// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test the case where we resolve `C::Result` and the trait bound
// itself includes a `Self::Item` shorthand.
//
// Regression test for issue #33425.

trait ParallelIterator {
    type Item;
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where C: Consumer<Self::Item>;
}

pub trait Consumer<ITEM> {
    type Result;
}

fn main() { }
