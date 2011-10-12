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
