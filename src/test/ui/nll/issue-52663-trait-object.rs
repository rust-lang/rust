// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(nll)]

trait Foo { fn get(&self); }

impl<A> Foo for A {
    fn get(&self) { }
}

fn main() {
    let _ = {
        let tmp0 = 3;
        let tmp1 = &tmp0;
        box tmp1 as Box<Foo + '_>
    };
    //~^^^ ERROR `tmp0` does not live long enough
}
