// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hashmap::HashSet;

struct Foo {
  n: HashSet<int>,
}

impl Foo {
    pub fn foo(&mut self, fun: |&int|) {
        for f in self.n.iter() {
            fun(f);
        }
    }
}

fn bar(f: &mut Foo) {
  f.foo(
        |a| { //~ ERROR closure requires unique access to `f`
            f.n.insert(*a);
        })
}

fn main() {
  let mut f = Foo { n: HashSet::new() };
  bar(&mut f);
}
