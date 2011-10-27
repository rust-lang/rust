use std;
import std::float;

#[test]
fn test_from_str() {
   assert ( float::from_str("3.14") == 3.14 );
   assert ( float::from_str("+3.14") == 3.14 );
   assert ( float::from_str("-3.14") == -3.14 );
   assert ( float::from_str("2.5E10") == 25000000000. );
   assert ( float::from_str("2.5e10") == 25000000000. );
   assert ( float::from_str("25000000000.E-10") == 2.5 );
   assert ( float::from_str("") == 0. );
   assert ( float::from_str("   ") == 0. );
   assert ( float::from_str(".") == 0. );
   assert ( float::from_str("5.") == 5. );
   assert ( float::from_str(".5") == 0.5 );
   assert ( float::from_str("0.5") == 0.5 );
}

#[test]
fn test_positive() {
  assert(float::positive(float::infinity()));
  assert(float::positive(1.));
  assert(float::positive(0.));
  assert(!float::positive(-1.));
  assert(!float::positive(float::neg_infinity()));
  assert(!float::positive(1./float::neg_infinity()));
}

#[test]
fn test_negative() {
  assert(!float::negative(float::infinity()));
  assert(!float::negative(1.));
  assert(!float::negative(0.));
  assert(float::negative(-1.));
  assert(float::negative(float::neg_infinity()));
  assert(float::negative(1./float::neg_infinity()));
}
