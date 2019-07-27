// run-pass
use std::f64;

pub fn main() {
  let nan: f64 = f64::NAN;
  assert!((nan).is_nan());

  let inf: f64 = f64::INFINITY;
  let neg_inf: f64 = -f64::INFINITY;
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
