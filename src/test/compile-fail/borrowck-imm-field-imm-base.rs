// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    x: uint
}

struct Bar {
    foo: Foo
}

fn main() {
    let mut b = Bar { foo: Foo { x: 3 } };
    let p = &b; //~ NOTE prior loan as immutable granted here
    let q = &mut b.foo.x; //~ ERROR loan of mutable local variable as mutable conflicts with prior loan
    let r = &p.foo.x;
    io::println(fmt!("*r = %u", *r));
    *q += 1;
    io::println(fmt!("*r = %u", *r));
}