// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate extra;

use std::num::Float;

pub fn main() {
  let nan: f64 = Float::nan();
  fail_unless!((nan).is_nan());

  let inf: f64 = Float::infinity();
  let neg_inf: f64 = Float::neg_infinity();
  assert_eq!(-inf, neg_inf);

  fail_unless!( nan !=  nan);
  fail_unless!( nan != -nan);
  fail_unless!(-nan != -nan);
  fail_unless!(-nan !=  nan);

  fail_unless!( nan !=   1.);
  fail_unless!( nan !=   0.);
  fail_unless!( nan !=  inf);
  fail_unless!( nan != -inf);

  fail_unless!(  1. !=  nan);
  fail_unless!(  0. !=  nan);
  fail_unless!( inf !=  nan);
  fail_unless!(-inf !=  nan);

  fail_unless!(!( nan == nan));
  fail_unless!(!( nan == -nan));
  fail_unless!(!( nan == 1.));
  fail_unless!(!( nan == 0.));
  fail_unless!(!( nan == inf));
  fail_unless!(!( nan == -inf));
  fail_unless!(!(  1. == nan));
  fail_unless!(!(  0. == nan));
  fail_unless!(!( inf == nan));
  fail_unless!(!(-inf == nan));
  fail_unless!(!(-nan == nan));
  fail_unless!(!(-nan == -nan));

  fail_unless!(!( nan >  nan));
  fail_unless!(!( nan > -nan));
  fail_unless!(!( nan >   0.));
  fail_unless!(!( nan >  inf));
  fail_unless!(!( nan > -inf));
  fail_unless!(!(  0. >  nan));
  fail_unless!(!( inf >  nan));
  fail_unless!(!(-inf >  nan));
  fail_unless!(!(-nan >  nan));

  fail_unless!(!(nan <   0.));
  fail_unless!(!(nan <   1.));
  fail_unless!(!(nan <  -1.));
  fail_unless!(!(nan <  inf));
  fail_unless!(!(nan < -inf));
  fail_unless!(!(nan <  nan));
  fail_unless!(!(nan < -nan));

  fail_unless!(!(  0. < nan));
  fail_unless!(!(  1. < nan));
  fail_unless!(!( -1. < nan));
  fail_unless!(!( inf < nan));
  fail_unless!(!(-inf < nan));
  fail_unless!(!(-nan < nan));

  fail_unless!((nan + inf).is_nan());
  fail_unless!((nan + -inf).is_nan());
  fail_unless!((nan + 0.).is_nan());
  fail_unless!((nan + 1.).is_nan());
  fail_unless!((nan * 1.).is_nan());
  fail_unless!((nan / 1.).is_nan());
  fail_unless!((nan / 0.).is_nan());
  fail_unless!((0.0/0.0f64).is_nan());
  fail_unless!((-inf + inf).is_nan());
  fail_unless!((inf - inf).is_nan());

  fail_unless!(!(-1.0f64).is_nan());
  fail_unless!(!(0.0f64).is_nan());
  fail_unless!(!(0.1f64).is_nan());
  fail_unless!(!(1.0f64).is_nan());
  fail_unless!(!(inf).is_nan());
  fail_unless!(!(-inf).is_nan());
  fail_unless!(!(1./-inf).is_nan());
}
