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
    mut x: uint
}

struct Bar {
    foo: Foo
}

fn main() {
    let mut b = Bar { foo: Foo { x: 3 } };
    let p = &b.foo.x;
    let q = &mut b.foo; //~ ERROR loan of mutable field as mutable conflicts with prior loan
    //~^ ERROR loan of mutable local variable as mutable conflicts with prior loan
    let r = &mut b; //~ ERROR loan of mutable local variable as mutable conflicts with prior loan
    //~^ ERROR loan of mutable local variable as mutable conflicts with prior loan
    io::println(fmt!("*p = %u", *p));
    q.x += 1;
    r.foo.x += 1;
    io::println(fmt!("*p = %u", *p));
}
