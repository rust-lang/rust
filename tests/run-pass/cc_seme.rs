// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(dead_code)]
enum Baz {
    One,
    Two,
}

struct Test {
    t: Option<usize>,
    b: Baz,
}

fn main() {}

pub fn foo() {
    use Baz::*;
    let x = Test { t: Some(0), b: One };

    match x {
        Test { t: Some(_), b: One } => unreachable!(),
        Test { t: Some(42), b: Two } => unreachable!(),
        Test { t: None, .. } => unreachable!(),
        Test { .. } => unreachable!(),
    }
}
