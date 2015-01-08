// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test FIXME(japari) remove test

struct Foo {
    f: for <'b> |&'b isize|:
      'b -> &'b isize //~ ERROR use of undeclared lifetime name `'b`
}

fn main() {
    let mut x: Vec< for <'a> ||
       :'a //~ ERROR use of undeclared lifetime name `'a`
    > = Vec::new();
    x.push(|| {});

    let foo = Foo {
        f: |x| x
    };
}
