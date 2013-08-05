// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn select<'r>(x: &'r Option<int>, y: &'r Option<int>) -> &'r Option<int> {
    match (x, y) {
        (&None, &None) => x,
        (&Some(_), _) => x,
        (&None, &Some(_)) => y
    }
}

pub fn main() {
    let x = None;
    let y = Some(3);
    assert_eq!(select(&x, &y).unwrap(), 3);
}
