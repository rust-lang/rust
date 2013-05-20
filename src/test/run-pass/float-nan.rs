// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;

use core::num::Float::{
  NaN, infinity, neg_infinity
};

pub fn main() {
  let nan = NaN::<float>();
  assert!((nan).is_NaN());

  let inf = infinity::<float>();
  assert_eq!(-inf, neg_infinity::<float>());

  assert!( nan !=  nan);
  assert!( nan != -nan);
  assert!(-nan != -nan);
  assert!(-nan !=  nan);

  assert!( nan !=   1.);
  assert!( nan !=   0.);
  assert!( nan !=  inf);
  assert!( nan != -inf);

  assert!(  1. !=  nan);
  assert!(  0. !=  nan);
  assert!( inf !=  nan);
  assert!(-inf !=  nan);

  assert!(!( nan == nan));
  assert!(!( nan == -nan));
  assert!(!( nan == 1.));
  assert!(!( nan == 0.));
  assert!(!( nan == inf));
  assert!(!( nan == -inf));
  assert!(!(  1. == nan));
  assert!(!(  0. == nan));
  assert!(!( inf == nan));
  assert!(!(-inf == nan));
  assert!(!(-nan == nan));
  assert!(!(-nan == -nan));

  assert!(!( nan >  nan));
  assert!(!( nan > -nan));
  assert!(!( nan >   0.));
  assert!(!( nan >  inf));
  assert!(!( nan > -inf));
  assert!(!(  0. >  nan));
  assert!(!( inf >  nan));
  assert!(!(-inf >  nan));
  assert!(!(-nan >  nan));

  assert!(!(nan <   0.));
  assert!(!(nan <   1.));
  assert!(!(nan <  -1.));
  assert!(!(nan <  inf));
  assert!(!(nan < -inf));
  assert!(!(nan <  nan));
  assert!(!(nan < -nan));

  assert!(!(  0. < nan));
  assert!(!(  1. < nan));
  assert!(!( -1. < nan));
  assert!(!( inf < nan));
  assert!(!(-inf < nan));
  assert!(!(-nan < nan));

  assert!((nan + inf).is_NaN());
  assert!((nan + -inf).is_NaN());
  assert!((nan + 0.).is_NaN());
  assert!((nan + 1.).is_NaN());
  assert!((nan * 1.).is_NaN());
  assert!((nan / 1.).is_NaN());
  assert!((nan / 0.).is_NaN());
  assert!((0f/0f).is_NaN());
  assert!((-inf + inf).is_NaN());
  assert!((inf - inf).is_NaN());

  assert!(!(-1f).is_NaN());
  assert!(!(0f).is_NaN());
  assert!(!(0.1f).is_NaN());
  assert!(!(1f).is_NaN());
  assert!(!(inf).is_NaN());
  assert!(!(-inf).is_NaN());
  assert!(!(1./-inf).is_NaN());
}
