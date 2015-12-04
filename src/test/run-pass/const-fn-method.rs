// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]

struct Foo { value: u32 }

impl Foo {
    const fn new() -> Foo {
        Foo { value: 22 }
    }
}

const FOO: Foo = Foo::new();

pub fn main() {
    assert_eq!(FOO.value, 22);
    let _: [&'static str; Foo::new().value as usize] = ["hey"; 22];
}
