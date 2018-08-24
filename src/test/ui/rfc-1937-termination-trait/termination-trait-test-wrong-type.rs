// compile-flags: --test

use std::num::ParseIntError;

#[test]
fn can_parse_zero_as_f32() -> Result<f32, ParseIntError> { //~ ERROR
    "0".parse()
}
