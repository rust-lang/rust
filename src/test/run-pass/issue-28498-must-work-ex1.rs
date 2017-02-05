// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Example taken from RFC 1238 text

// https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md
//     #examples-of-code-that-must-continue-to-work

use std::cell::Cell;

struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);

fn main() {
    let mut data = Vec::new();
    data.push(Concrete(0, Cell::new(None)));
    data.push(Concrete(0, Cell::new(None)));

    data[0].1.set(Some(&data[1]));
    data[1].1.set(Some(&data[0]));
}
