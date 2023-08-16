//@run
// exec-env:TEST_EXEC_ENV=22
//@ignore-target-emscripten FIXME: issue #31622
//@ignore-target-sgx unsupported

use std::env;

pub fn main() {
    assert_eq!(env::var("TEST_EXEC_ENV"), Ok("22".to_string()));
}
