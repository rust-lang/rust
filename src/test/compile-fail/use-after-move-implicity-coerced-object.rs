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

use std::fmt;

struct Number {
    n: i64
}

impl fmt::Show for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, "{}", self.n)
    }
}

struct List {
    list: Vec<~ToStr> }

impl List {
    fn push(&mut self, n: ~ToStr) {
        self.list.push(n);
    }
}

fn main() {
    let n = ~Number { n: 42 };
    let mut l = ~List { list: Vec::new() };
    l.push(n);
    let x = n.to_str();
    //~^ ERROR: use of moved value: `n`
}
