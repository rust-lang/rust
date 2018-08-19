// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![deny(warnings)]

use std::collections::BTreeMap;

pub struct RangeMap {
    map: BTreeMap<Range, u8>,
}

#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Range;

impl RangeMap {
    fn iter_with_range<'a>(&'a self) -> impl Iterator<Item = (&'a Range, &'a u8)> + 'a {
        self.map.range(Range..Range)
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a u8> + 'a {
        self.iter_with_range().map(|(_, data)| data)
    }

}

fn main() {}
