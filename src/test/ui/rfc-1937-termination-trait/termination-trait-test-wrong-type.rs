// compile-flags: --test
// ignore-x86 FIXME: missing sysroot spans (#53081)

use std::num::ParseFloatError;

#[test]
fn can_parse_zero_as_f32() -> Result<f32, ParseFloatError> { //~ ERROR
    "0".parse()
}
