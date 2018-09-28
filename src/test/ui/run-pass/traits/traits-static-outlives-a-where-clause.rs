// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

trait Foo<'a> {
    fn xyz(self);
}
impl<'a, T> Foo<'a> for T where 'static: 'a {
    fn xyz(self) {}
}

trait Bar {
    fn uvw(self);
}
impl<T> Bar for T where for<'a> T: Foo<'a> {
    fn uvw(self) { self.xyz(); }
}

fn foo<T>(t: T) where T: Bar {
    t.uvw();
}

fn main() {
    foo(0);
}
