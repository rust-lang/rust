// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that when you use ufcs form to invoke a trait method (on a
// trait object) everything works fine.

// pretty-expanded FIXME #23616

trait Foo {
    fn test(&self) -> i32;
}

impl Foo for i32 {
    fn test(&self) -> i32 { *self }
}

fn main() {
    let a: &Foo = &22;
    assert_eq!(Foo::test(a), 22);
}
