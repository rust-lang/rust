// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test associated type references in structure fields.

trait Test {
    type V;

    fn test(&self, value: &Self::V) -> bool;
}

///////////////////////////////////////////////////////////////////////////

struct TesterPair<T:Test> {
    tester: T,
    value: T::V,
}

impl<T:Test> TesterPair<T> {
    fn new(tester: T, value: T::V) -> TesterPair<T> {
        TesterPair { tester: tester, value: value }
    }

    fn test(&self) -> bool {
        self.tester.test(&self.value)
    }
}

///////////////////////////////////////////////////////////////////////////

struct EqU32(u32);
impl Test for EqU32 {
    type V = u32;

    fn test(&self, value: &u32) -> bool {
        self.0 == *value
    }
}

struct EqI32(i32);
impl Test for EqI32 {
    type V = i32;

    fn test(&self, value: &i32) -> bool {
        self.0 == *value
    }
}

fn main() {
    let tester = TesterPair::new(EqU32(22), 23);
    tester.test();

    let tester = TesterPair::new(EqI32(22), 23);
    tester.test();
}
