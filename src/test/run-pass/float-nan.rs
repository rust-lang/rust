// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num::Float;

pub fn main() {
  let nan: f64 = Float::nan();
  assert!((nan).is_nan());

  let inf: f64 = Float::infinity();
  let neg_inf: f64 = Float::neg_infinity();
  assert_eq!(-inf, neg_inf);

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

  assert!((nan + inf).is_nan());
  assert!((nan + -inf).is_nan());
  assert!((nan + 0.).is_nan());
  assert!((nan + 1.).is_nan());
  assert!((nan * 1.).is_nan());
  assert!((nan / 1.).is_nan());
  assert!((nan / 0.).is_nan());
  assert!((0.0/0.0f64).is_nan());
  assert!((-inf + inf).is_nan());
  assert!((inf - inf).is_nan());

  assert!(!(-1.0f64).is_nan());
  assert!(!(0.0f64).is_nan());
  assert!(!(0.1f64).is_nan());
  assert!(!(1.0f64).is_nan());
  assert!(!(inf).is_nan());
  assert!(!(-inf).is_nan());
  assert!(!(1./-inf).is_nan());
}
