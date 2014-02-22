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

fn test_nil() {
    fail_unless_eq!((), ());
    fail_unless!((!(() != ())));
    fail_unless!((!(() < ())));
    fail_unless!((() <= ()));
    fail_unless!((!(() > ())));
    fail_unless!((() >= ()));
}

fn test_bool() {
    fail_unless!((!(true < false)));
    fail_unless!((!(true <= false)));
    fail_unless!((true > false));
    fail_unless!((true >= false));

    fail_unless!((false < true));
    fail_unless!((false <= true));
    fail_unless!((!(false > true)));
    fail_unless!((!(false >= true)));

    // Bools support bitwise binops
    fail_unless_eq!(false & false, false);
    fail_unless_eq!(true & false, false);
    fail_unless_eq!(true & true, true);
    fail_unless_eq!(false | false, false);
    fail_unless_eq!(true | false, true);
    fail_unless_eq!(true | true, true);
    fail_unless_eq!(false ^ false, false);
    fail_unless_eq!(true ^ false, true);
    fail_unless_eq!(true ^ true, false);
}

fn test_box() {
    fail_unless_eq!(@10, @10);
}

fn test_ptr() {
    unsafe {
        let p1: *u8 = ::std::cast::transmute(0);
        let p2: *u8 = ::std::cast::transmute(0);
        let p3: *u8 = ::std::cast::transmute(1);

        fail_unless_eq!(p1, p2);
        fail_unless!(p1 != p3);
        fail_unless!(p1 < p3);
        fail_unless!(p1 <= p3);
        fail_unless!(p3 > p1);
        fail_unless!(p3 >= p3);
        fail_unless!(p1 <= p2);
        fail_unless!(p1 >= p2);
    }
}

#[deriving(Eq)]
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
  error!("q = {:x}, r = {:x}",
         (::std::cast::transmute::<*p, uint>(&q)),
         (::std::cast::transmute::<*p, uint>(&r)));
  }
  fail_unless_eq!(q, r);
  r.y = 17;
  fail_unless!((r.y != q.y));
  fail_unless_eq!(r.y, 17);
  fail_unless!((q != r));
}

pub fn main() {
    test_nil();
    test_bool();
    test_box();
    test_ptr();
    test_class();
}
