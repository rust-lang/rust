// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Binop corner cases

#![feature(managed_boxes)]

fn test_nil() {
    assert_eq!((), ());
    assert!((!(() != ())));
    assert!((!(() < ())));
    assert!((() <= ()));
    assert!((!(() > ())));
    assert!((() >= ()));
}

fn test_bool() {
    assert!((!(true < false)));
    assert!((!(true <= false)));
    assert!((true > false));
    assert!((true >= false));

    assert!((false < true));
    assert!((false <= true));
    assert!((!(false > true)));
    assert!((!(false >= true)));

    // Bools support bitwise binops
    assert_eq!(false & false, false);
    assert_eq!(true & false, false);
    assert_eq!(true & true, true);
    assert_eq!(false | false, false);
    assert_eq!(true | false, true);
    assert_eq!(true | true, true);
    assert_eq!(false ^ false, false);
    assert_eq!(true ^ false, true);
    assert_eq!(true ^ true, false);
}

fn test_box() {
    assert_eq!(@10, @10);
}

fn test_ptr() {
    unsafe {
        let p1: *u8 = ::std::mem::transmute(0);
        let p2: *u8 = ::std::mem::transmute(0);
        let p3: *u8 = ::std::mem::transmute(1);

        assert_eq!(p1, p2);
        assert!(p1 != p3);
        assert!(p1 < p3);
        assert!(p1 <= p3);
        assert!(p3 > p1);
        assert!(p3 >= p3);
        assert!(p1 <= p2);
        assert!(p1 >= p2);
    }
}

#[deriving(PartialEq, Show)]
struct p {
  x: int,
  y: int,
}

fn p(x: int, y: int) -> p {
    p {
        x: x,
        y: y
    }
}

fn test_class() {
  let q = p(1, 2);
  let mut r = p(1, 2);

  unsafe {
  println!("q = {:x}, r = {:x}",
         (::std::mem::transmute::<*p, uint>(&q)),
         (::std::mem::transmute::<*p, uint>(&r)));
  }
  assert_eq!(q, r);
  r.y = 17;
  assert!((r.y != q.y));
  assert_eq!(r.y, 17);
  assert!((q != r));
}

pub fn main() {
    test_nil();
    test_bool();
    test_box();
    test_ptr();
    test_class();
}
