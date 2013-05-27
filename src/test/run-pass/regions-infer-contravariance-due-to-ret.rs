// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct boxed_int<'self> {
    f: &'self int,
}

fn max<'r>(bi: &'r boxed_int, f: &'r int) -> int {
    if *bi.f > *f {*bi.f} else {*f}
}

fn with(bi: &boxed_int) -> int {
    let i = 22;
    max(bi, &i)
}

pub fn main() {
    let g = 21;
    let foo = boxed_int { f: &g };
    assert_eq!(with(&foo), 22);
}
