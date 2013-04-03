// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::hashmap::HashSet;

struct Foo {
  n: HashSet<int>,
}

pub impl Foo {
  fn foo(&mut self, fun: &fn(&int)) {
    for self.n.each |f| {
      fun(f);
    }
  }
}

fn bar(f: &mut Foo) {
  do f.foo |a| { //~ NOTE prior loan as mutable granted here
    f.n.insert(*a); //~ ERROR conflicts with prior loan
  }
}

fn main() {
  let mut f = Foo { n: HashSet::new() };
  bar(&mut f);
}
