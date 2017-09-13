// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a field can have the same name in different variants
// of an enum
// FIXME #27889

pub enum Foo {
    X { foo: u32 },
    Y { foo: u32 }
}

pub fn foo(mut x: Foo) {
    let mut y = None;
    let mut z = None;
    if let Foo::X { ref foo } = x {
        z = Some(foo);
    }
    if let Foo::Y { ref mut foo } = x {
        y = Some(foo);
    }
}

fn main() {}
