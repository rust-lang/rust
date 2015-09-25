// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A method's receiver must be well-formed, even if it has late-bound regions.
// Because of this, a method's substs being well-formed does not imply that
// the method's implied bounds are met.

struct Foo<'b>(Option<&'b ()>);

trait Bar<'b> {
    fn xmute<'a>(&'a self, u: &'b u32) -> &'a u32;
}

impl<'b> Bar<'b> for Foo<'b> {
    fn xmute<'a>(&'a self, u: &'b u32) -> &'a u32 { u }
}

fn main() {
    let f = Foo(None);
    let f2 = f;
    let dangling = {
        let pointer = Box::new(42);
        f2.xmute(&pointer) //~ ERROR `pointer` does not live long enough
    };
    println!("{}", dangling);
}
