// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct X {
    repr: int
}

fn apply<T>(x: T, f: &fn(T)) {
    f(x);
}

fn check_int(x: int) {
    assert_eq!(x, 22);
}

fn check_struct(x: X) {
    check_int(x.repr);
}

pub fn main() {
    apply(22, check_int);
    apply(X {repr: 22}, check_struct);
}
