// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! higher_order {
    (subst $lhs:tt => $rhs:tt) => ({
            macro_rules! anon { $lhs => $rhs }
            anon!(1_usize, 2_usize, "foo")
    });
}

fn main() {
    let val = higher_order!(subst ($x:expr, $y:expr, $foo:expr) => (($x + $y, $foo)));
    assert_eq!(val, (3, "foo"));
}
