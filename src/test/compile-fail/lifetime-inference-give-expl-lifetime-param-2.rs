// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

use std::iter::{Range,range};

trait Itble<'r, T, I: Iterator<T>> { fn iter(&'r self) -> I; }

impl<'r> Itble<'r, uint, Range<uint>> for (uint, uint) {
    fn iter(&'r self) -> Range<uint> {
        let &(min, max) = self;
        range(min, max)
    }
}

fn check<'r, I: Iterator<uint>, T: Itble<'r, uint, I>>(cont: &T) -> bool {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn check<'r, I: Iterator<uint>, T: Itble<'r, uint, I>>(cont: &'r T) -> bool
    let cont_iter = cont.iter(); //~ ERROR: cannot infer
    let result = cont_iter.fold(Some(0u16), |state, val| {
        state.map_or(None, |mask| {
            let bit = 1 << val;
            if mask & bit == 0 {Some(mask|bit)} else {None}
        })
    });
    result.is_some()
}

fn main() {}
