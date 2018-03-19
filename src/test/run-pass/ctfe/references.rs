// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const FOO: &[u8] = b"foo";
const BAR: &[u8] = &[1, 2, 3];

const BOO: &i32 = &42;

fn main() {
    match &[1u8, 2, 3] as &[u8] {
        FOO => panic!("a"),
        BAR => println!("b"),
        _ => panic!("c"),
    }

    match b"foo" as &[u8] {
        FOO => println!("a"),
        BAR => panic!("b"),
        _ => panic!("c"),
    }

    match &43 {
        &42 => panic!(),
        BOO => panic!(),
        _ => println!("d"),
    }
}
