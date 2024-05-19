//@ compile-flags: --test

use std::num::ParseFloatError;

#[test]
fn can_parse_zero_as_f32() -> Result<f32, ParseFloatError> { //~ ERROR
    "0".parse()
}
