// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that we are able to distinguish when loans borrow different
// anonymous fields of an enum variant vs the same anonymous field.

enum Foo {
    X, Y(uint, uint)
}

fn distinct_variant() {
    let mut y = Y(1, 2);

    let a = match y {
      Y(ref mut a, _) => a,
      X => fail!()
    };

    let b = match y {
      Y(_, ref mut b) => b,
      X => fail!()
    };

    *a += 1;
    *b += 1;
}

fn same_variant() {
    let mut y = Y(1, 2);

    let a = match y {
      Y(ref mut a, _) => a,
      X => fail!()
    };

    let b = match y {
      Y(ref mut b, _) => b, //~ ERROR cannot borrow
      X => fail!()
    };

    *a += 1;
    *b += 1;
}

fn main() {
}
