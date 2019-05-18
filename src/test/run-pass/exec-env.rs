// exec-env:TEST_EXEC_ENV=22
// ignore-cloudabi no env vars
// ignore-emscripten FIXME: issue #31622
// ignore-sgx unsupported

use std::env;

pub fn main() {
    assert_eq!(env::var("TEST_EXEC_ENV"), Ok("22".to_string()));
}
