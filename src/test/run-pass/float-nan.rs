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

pub fn main() {
  let nan = float::NaN;
  fail_unless!((float::is_NaN(nan)));

  let inf = float::infinity;
  fail_unless!((-inf == float::neg_infinity));

  fail_unless!(( nan !=  nan));
  fail_unless!(( nan != -nan));
  fail_unless!((-nan != -nan));
  fail_unless!((-nan !=  nan));

  fail_unless!(( nan !=   1.));
  fail_unless!(( nan !=   0.));
  fail_unless!(( nan !=  inf));
  fail_unless!(( nan != -inf));

  fail_unless!((  1. !=  nan));
  fail_unless!((  0. !=  nan));
  fail_unless!(( inf !=  nan));
  fail_unless!((-inf !=  nan));

  fail_unless!((!( nan ==  nan)));
  fail_unless!((!( nan == -nan)));
  fail_unless!((!( nan ==   1.)));
  fail_unless!((!( nan ==   0.)));
  fail_unless!((!( nan ==  inf)));
  fail_unless!((!( nan == -inf)));
  fail_unless!((!(  1. ==  nan)));
  fail_unless!((!(  0. ==  nan)));
  fail_unless!((!( inf ==  nan)));
  fail_unless!((!(-inf ==  nan)));
  fail_unless!((!(-nan ==  nan)));
  fail_unless!((!(-nan == -nan)));

  fail_unless!((!( nan >  nan)));
  fail_unless!((!( nan > -nan)));
  fail_unless!((!( nan >   0.)));
  fail_unless!((!( nan >  inf)));
  fail_unless!((!( nan > -inf)));
  fail_unless!((!(  0. >  nan)));
  fail_unless!((!( inf >  nan)));
  fail_unless!((!(-inf >  nan)));
  fail_unless!((!(-nan >  nan)));

  fail_unless!((!(nan <   0.)));
  fail_unless!((!(nan <   1.)));
  fail_unless!((!(nan <  -1.)));
  fail_unless!((!(nan <  inf)));
  fail_unless!((!(nan < -inf)));
  fail_unless!((!(nan <  nan)));
  fail_unless!((!(nan < -nan)));

  fail_unless!((!(  0. < nan)));
  fail_unless!((!(  1. < nan)));
  fail_unless!((!( -1. < nan)));
  fail_unless!((!( inf < nan)));
  fail_unless!((!(-inf < nan)));
  fail_unless!((!(-nan < nan)));

  fail_unless!((float::is_NaN(nan + inf)));
  fail_unless!((float::is_NaN(nan + -inf)));
  fail_unless!((float::is_NaN(nan + 0.)));
  fail_unless!((float::is_NaN(nan + 1.)));
  fail_unless!((float::is_NaN(nan * 1.)));
  fail_unless!((float::is_NaN(nan / 1.)));
  fail_unless!((float::is_NaN(nan / 0.)));
  fail_unless!((float::is_NaN(0. / 0.)));
  fail_unless!((float::is_NaN(-inf + inf)));
  fail_unless!((float::is_NaN(inf - inf)));

  fail_unless!((!float::is_NaN(-1.)));
  fail_unless!((!float::is_NaN(0.)));
  fail_unless!((!float::is_NaN(0.1)));
  fail_unless!((!float::is_NaN(1.)));
  fail_unless!((!float::is_NaN(inf)));
  fail_unless!((!float::is_NaN(-inf)));
  fail_unless!((!float::is_NaN(1./-inf)));
}
