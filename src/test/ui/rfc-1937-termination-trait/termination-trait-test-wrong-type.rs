// compile-flags: --test
// ignore-x86
// ^ due to stderr output differences

use std::num::ParseFloatError;

#[test]
fn can_parse_zero_as_f32() -> Result<f32, ParseFloatError> { //~ ERROR
    "0".parse()
}
