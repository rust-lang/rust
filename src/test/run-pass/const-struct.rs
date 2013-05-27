// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct foo { a: int, b: int, c: int }

impl cmp::Eq for foo {
    fn eq(&self, other: &foo) -> bool {
        (*self).a == (*other).a &&
        (*self).b == (*other).b &&
        (*self).c == (*other).c
    }
    fn ne(&self, other: &foo) -> bool { !(*self).eq(other) }
}

static x : foo = foo { a:1, b:2, c: 3 };
static y : foo = foo { b:2, c:3, a: 1 };
static z : &'static foo = &foo { a: 10, b: 22, c: 12 };

pub fn main() {
    assert_eq!(x.b, 2);
    assert_eq!(x, y);
    assert_eq!(z.b, 22);
    io::println(fmt!("0x%x", x.b as uint));
    io::println(fmt!("0x%x", z.c as uint));
}
