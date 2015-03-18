// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug)]
enum Foo { One, Two, Three }
use Foo::*;

fn inc(x: Foo) -> Foo {
    match x {
        One => Two,
        Two => Three,
        Three => Three,
    }
}

pub fn main() {
    let mut x = Some(2);
    while let Some(1) | Some(2) = x {
        x = None;
    }

    assert_eq!(x , None);

    let mut x = One;
    let mut n = 0;
    while let One | Two = x {
        n += 1;
        x = inc(x);
    }

    assert_eq!(n, 2);
}
