// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can use method notation to call methods based on a
// where clause type, and not only type parameters.

// pretty-expanded FIXME #23616

trait Foo {
    fn foo(&self) -> i32;
}

impl Foo for Option<i32>
{
    fn foo(&self) -> i32 {
        self.unwrap_or(22)
    }
}

impl Foo for Option<u32>
{
    fn foo(&self) -> i32 {
        self.unwrap_or(22) as i32
    }
}

fn check<T>(x: Option<T>) -> (i32, i32)
    where Option<T> : Foo
{
    let y: Option<T> = None;
    (x.foo(), y.foo())
}

fn main() {
    assert_eq!(check(Some(23u32)), (23, 22));
    assert_eq!(check(Some(23)), (23, 22));
}
