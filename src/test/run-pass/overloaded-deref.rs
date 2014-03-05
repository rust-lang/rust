// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::rc::Rc;

#[deriving(Eq, Show)]
struct Point {
    x: int,
    y: int
}

pub fn main() {
    assert_eq!(*Rc::new(5), 5);
    assert_eq!(***Rc::new(~~5), 5);
    assert_eq!(*Rc::new(Point {x: 2, y: 4}), Point {x: 2, y: 4});

    let i = Rc::new(RefCell::new(2));
    let i_value = *(*i).borrow();
    *(*i).borrow_mut() = 5;
    assert_eq!((i_value, *(*i).borrow()), (2, 5));

    let s = Rc::new(~"foo");
    assert_eq!(*s, ~"foo");
    assert_eq!((*s).as_slice(), "foo");

    let mut_s = Rc::new(RefCell::new(~"foo"));
    (*(*mut_s).borrow_mut()).push_str("bar");
    // assert_eq! would fail here because it stores the LHS and RHS in two locals.
    assert!((*(*mut_s).borrow()).as_slice() == "foobar");
    assert!((*(*mut_s).borrow_mut()).as_slice() == "foobar");

    let p = Rc::new(RefCell::new(Point {x: 1, y: 2}));
    (*(*p).borrow_mut()).x = 3;
    (*(*p).borrow_mut()).y += 3;
    assert_eq!(*(*p).borrow(), Point {x: 3, y: 5});

    let v = Rc::new(RefCell::new(~[1, 2, 3]));
    (*(*v).borrow_mut())[0] = 3;
    (*(*v).borrow_mut())[1] += 3;
    assert_eq!(((*(*v).borrow())[0], (*(*v).borrow())[1], (*(*v).borrow())[2]), (3, 5, 3));
}
