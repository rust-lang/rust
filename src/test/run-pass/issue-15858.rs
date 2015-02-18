// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsafe_destructor)]

static mut DROP_RAN: bool = false;

trait Bar {
    fn do_something(&mut self);
}

struct BarImpl;

impl Bar for BarImpl {
    fn do_something(&mut self) {}
}


struct Foo<B>(B);

#[unsafe_destructor]
impl<B: Bar> Drop for Foo<B> {
    fn drop(&mut self) {
        unsafe {
            DROP_RAN = true;
        }
    }
}


fn main() {
    {
       let _x: Foo<BarImpl> = Foo(BarImpl);
    }
    unsafe {
        assert_eq!(DROP_RAN, true);
    }
}
