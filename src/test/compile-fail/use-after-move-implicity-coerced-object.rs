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

struct Number {
    n: i64
}

impl ToStr for Number {
    fn to_str(&self) -> ~str {
        self.n.to_str()
    }
}

struct List {
    list: ~[~ToStr]
}

impl List {
    fn push(&mut self, n: ~ToStr) {
        self.list.push(n);
    }
}

fn main() {
    let n = ~Number { n: 42 };
    let mut l = ~List { list: ~[] };
    l.push(n);
    let x = n.to_str();
    //~^ ERROR: use of moved value: `n`
}
