// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;
use std::gc::{GC, Gc};
use std::cell::RefCell;

static mut DROPS: uint = 0;

struct Foo(bool);
impl Drop for Foo {
    fn drop(&mut self) {
        let Foo(fail) = *self;
        unsafe { DROPS += 1; }
        if fail { fail!() }
    }
}

fn tld_fail(fail: bool) {
    local_data_key!(foo: Foo);
    foo.replace(Some(Foo(fail)));
}

fn gc_fail(fail: bool) {
    struct A {
        inner: RefCell<Option<Gc<A>>>,
        other: Foo,
    }
    let a = box(GC) A {
        inner: RefCell::new(None),
        other: Foo(fail),
    };
    *a.inner.borrow_mut() = Some(a.clone());
}

fn main() {
    let _ = task::try(proc() {
        tld_fail(true);
        gc_fail(false);
    });

    unsafe {
        assert_eq!(DROPS, 2);
    }
}

