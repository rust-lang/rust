// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test method calls with self as an argument cannot subvert borrow checking.

struct Foo;

impl Foo {
    fn bar(&self) {}
    fn baz(&mut self) {}
}

fn main() {
    let mut x = Foo;
    let y = &mut x;
    Foo::bar(&x); //~ERROR cannot borrow `x`

    let x = Foo;
    Foo::baz(&x); //~ERROR cannot borrow immutable dereference of `&`-pointer as mutable
}
