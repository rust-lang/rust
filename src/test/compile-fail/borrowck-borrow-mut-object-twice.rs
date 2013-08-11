// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that `&mut` objects cannot be borrowed twice, just like
// other `&mut` pointers.

trait Foo {
    fn f1<'a>(&'a mut self) -> &'a ();
    fn f2(&mut self);
}

fn test(x: &mut Foo) {
    let _y = x.f1();
    x.f2(); //~ ERROR cannot borrow `*x` as mutable more than once at a time
}

fn main() {}
