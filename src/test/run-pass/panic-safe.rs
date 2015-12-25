// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![feature(recover)]

use std::panic::{RecoverSafe, AssertRecoverSafe};
use std::cell::RefCell;
use std::sync::{Mutex, RwLock, Arc};
use std::rc::Rc;

struct Foo { a: i32 }

fn assert<T: RecoverSafe + ?Sized>() {}

fn main() {
    assert::<i32>();
    assert::<&i32>();
    assert::<*mut i32>();
    assert::<*const i32>();
    assert::<usize>();
    assert::<str>();
    assert::<&str>();
    assert::<Foo>();
    assert::<&Foo>();
    assert::<Vec<i32>>();
    assert::<String>();
    assert::<RefCell<i32>>();
    assert::<Box<i32>>();
    assert::<Mutex<i32>>();
    assert::<RwLock<i32>>();
    assert::<Rc<i32>>();
    assert::<Arc<i32>>();

    fn bar<T>() {
        assert::<Mutex<T>>();
        assert::<RwLock<T>>();
    }
    fn baz<T: RecoverSafe>() {
        assert::<Box<T>>();
        assert::<Vec<T>>();
        assert::<RefCell<T>>();
        assert::<AssertRecoverSafe<T>>();
        assert::<&AssertRecoverSafe<T>>();
        assert::<Rc<AssertRecoverSafe<T>>>();
        assert::<Arc<AssertRecoverSafe<T>>>();
    }
}
