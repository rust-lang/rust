// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Test for `boxed` mod.

use core::any::Any;
use core::ops::Deref;
use core::result::Result::{Ok, Err};
use core::clone::Clone;

use std::boxed::Box;
use std::boxed::BoxAny;

#[test]
fn test_owned_clone() {
    let a = Box::new(5i);
    let b: Box<int> = a.clone();
    assert!(a == b);
}

#[derive(PartialEq, Eq)]
struct Test;

#[test]
fn any_move() {
    let a = Box::new(8u) as Box<Any>;
    let b = Box::new(Test) as Box<Any>;

    match a.downcast::<uint>() {
        Ok(a) => { assert!(a == Box::new(8u)); }
        Err(..) => panic!()
    }
    match b.downcast::<Test>() {
        Ok(a) => { assert!(a == Box::new(Test)); }
        Err(..) => panic!()
    }

    let a = Box::new(8u) as Box<Any>;
    let b = Box::new(Test) as Box<Any>;

    assert!(a.downcast::<Box<Test>>().is_err());
    assert!(b.downcast::<Box<uint>>().is_err());
}

#[test]
fn test_show() {
    let a = Box::new(8u) as Box<Any>;
    let b = Box::new(Test) as Box<Any>;
    let a_str = format!("{:?}", a);
    let b_str = format!("{:?}", b);
    assert_eq!(a_str, "Box<Any>");
    assert_eq!(b_str, "Box<Any>");

    static EIGHT: usize = 8us;
    static TEST: Test = Test;
    let a = &EIGHT as &Any;
    let b = &TEST as &Any;
    let s = format!("{:?}", a);
    assert_eq!(s, "&Any");
    let s = format!("{:?}", b);
    assert_eq!(s, "&Any");
}

#[test]
fn deref() {
    fn homura<T: Deref<Target=i32>>(_: T) { }
    homura(Box::new(765i32));
}
