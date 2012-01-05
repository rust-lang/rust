import core::*;

use std;
import float;

#[test]
fn test_from_str() {
   assert ( float::from_str("3") == 3. );
   assert ( float::from_str("  3  ") == 3. );
   assert ( float::from_str("3.14") == 3.14 );
   assert ( float::from_str("+3.14") == 3.14 );
   assert ( float::from_str("-3.14") == -3.14 );
   assert ( float::from_str("2.5E10") == 25000000000. );
   assert ( float::from_str("2.5e10") == 25000000000. );
   assert ( float::from_str("25000000000.E-10") == 2.5 );
   assert ( float::from_str("") == 0. );
   assert ( float::from_str(".") == 0. );
   assert ( float::from_str(".e1") == 0. );
   assert ( float::from_str(".e-1") == 0. );
   assert ( float::from_str("5.") == 5. );
   assert ( float::from_str(".5") == 0.5 );
   assert ( float::from_str("0.5") == 0.5 );
   assert ( float::from_str("0.5 ") == 0.5 );
   assert ( float::from_str(" 0.5 ") == 0.5 );
   assert ( float::from_str(" -.5 ") == -0.5 );
   assert ( float::from_str(" -.5 ") == -0.5 );
   assert ( float::from_str(" -5 ") == -5. );

   assert ( float::is_NaN(float::from_str("x")) );
   assert ( float::from_str(" ") == 0. );
   assert ( float::from_str("   ") == 0. );
   assert ( float::from_str(" 0.5") == 0.5 );
   assert ( float::from_str(" 0.5 ") == 0.5 );
   assert ( float::from_str(" .1 ") == 0.1 );
   assert ( float::is_NaN(float::from_str("e")) );
   assert ( float::is_NaN(float::from_str("E")) );
   assert ( float::is_NaN(float::from_str("E1")) );
   assert ( float::is_NaN(float::from_str("1e1e1")) );
   assert ( float::is_NaN(float::from_str("1e1.1")) );
   assert ( float::is_NaN(float::from_str("1e1-1")) );
}

#[test]
fn test_positive() {
  assert(float::is_positive(float::infinity));
  assert(float::is_positive(1.));
  assert(float::is_positive(0.));
  assert(!float::is_positive(-1.));
  assert(!float::is_positive(float::neg_infinity));
  assert(!float::is_positive(1./float::neg_infinity));
  assert(!float::is_positive(float::NaN));
}

#[test]
fn test_negative() {
  assert(!float::is_negative(float::infinity));
  assert(!float::is_negative(1.));
  assert(!float::is_negative(0.));
  assert(float::is_negative(-1.));
  assert(float::is_negative(float::neg_infinity));
  assert(float::is_negative(1./float::neg_infinity));
  assert(!float::is_negative(float::NaN));
}

#[test]
fn test_nonpositive() {
  assert(!float::is_nonpositive(float::infinity));
  assert(!float::is_nonpositive(1.));
  assert(!float::is_nonpositive(0.));
  assert(float::is_nonpositive(-1.));
  assert(float::is_nonpositive(float::neg_infinity));
  assert(float::is_nonpositive(1./float::neg_infinity));
  assert(!float::is_nonpositive(float::NaN));
}

#[test]
fn test_nonnegative() {
  assert(float::is_nonnegative(float::infinity));
  assert(float::is_nonnegative(1.));
  assert(float::is_nonnegative(0.));
  assert(!float::is_nonnegative(-1.));
  assert(!float::is_nonnegative(float::neg_infinity));
  assert(!float::is_nonnegative(1./float::neg_infinity));
  assert(!float::is_nonnegative(float::NaN));
}
