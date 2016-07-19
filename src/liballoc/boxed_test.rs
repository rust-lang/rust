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
use core::result::Result::{Err, Ok};
use core::clone::Clone;

use std::boxed::Box;

#[test]
fn test_owned_clone() {
    let a = Box::new(5);
    let b: Box<i32> = a.clone();
    assert!(a == b);
}

#[derive(PartialEq, Eq)]
struct Test;

#[test]
fn any_move() {
    let a = Box::new(8) as Box<Any>;
    let b = Box::new(Test) as Box<Any>;

    match a.downcast::<i32>() {
        Ok(a) => {
            assert!(a == Box::new(8));
        }
        Err(..) => panic!(),
    }
    match b.downcast::<Test>() {
        Ok(a) => {
            assert!(a == Box::new(Test));
        }
        Err(..) => panic!(),
    }

    let a = Box::new(8) as Box<Any>;
    let b = Box::new(Test) as Box<Any>;

    assert!(a.downcast::<Box<Test>>().is_err());
    assert!(b.downcast::<Box<i32>>().is_err());
}

#[test]
fn test_show() {
    let a = Box::new(8) as Box<Any>;
    let b = Box::new(Test) as Box<Any>;
    let a_str = format!("{:?}", a);
    let b_str = format!("{:?}", b);
    assert_eq!(a_str, "Any");
    assert_eq!(b_str, "Any");

    static EIGHT: usize = 8;
    static TEST: Test = Test;
    let a = &EIGHT as &Any;
    let b = &TEST as &Any;
    let s = format!("{:?}", a);
    assert_eq!(s, "Any");
    let s = format!("{:?}", b);
    assert_eq!(s, "Any");
}

#[test]
fn deref() {
    fn homura<T: Deref<Target = i32>>(_: T) {}
    homura(Box::new(765));
}

#[test]
fn raw_sized() {
    let x = Box::new(17);
    let p = Box::into_raw(x);
    unsafe {
        assert_eq!(17, *p);
        *p = 19;
        let y = Box::from_raw(p);
        assert_eq!(19, *y);
    }
}

#[test]
fn raw_trait() {
    trait Foo {
        fn get(&self) -> u32;
        fn set(&mut self, value: u32);
    }

    struct Bar(u32);

    impl Foo for Bar {
        fn get(&self) -> u32 {
            self.0
        }

        fn set(&mut self, value: u32) {
            self.0 = value;
        }
    }

    let x: Box<Foo> = Box::new(Bar(17));
    let p = Box::into_raw(x);
    unsafe {
        assert_eq!(17, (*p).get());
        (*p).set(19);
        let y: Box<Foo> = Box::from_raw(p);
        assert_eq!(19, y.get());
    }
}
