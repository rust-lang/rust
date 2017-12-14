// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(arbitrary_self_types)]

use std::rc::Rc;

struct Foo(String);

impl Foo {
    unsafe fn foo(self: *const Self) -> *const str {
        (*self).0.as_ref()
    }

    fn complicated_1(self: *const Rc<Self>) -> &'static str {
        "Foo::complicated_1"
    }

    unsafe fn complicated_2(self: Rc<*const Self>) -> *const str {
        (**self).0.as_ref()
    }
}

fn main() {
    let foo = Foo("abc123".into());
    assert_eq!("abc123", unsafe { &*(&foo as *const Foo).foo() });
    assert_eq!("Foo::complicated_1", std::ptr::null::<Rc<Foo>>().complicated_1());
    let rc = Rc::new(&foo as *const Foo);
    assert_eq!("abc123", unsafe { &*rc.complicated_2()});
}
