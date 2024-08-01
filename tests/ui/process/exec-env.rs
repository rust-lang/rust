//@ run-pass
//@ exec-env:TEST_EXEC_ENV=22
//@ ignore-wasm32 wasm runtimes aren't configured to inherit env vars yet
//@ ignore-sgx unsupported

use std::env;

pub fn main() {
    assert_eq!(env::var("TEST_EXEC_ENV"), Ok("22".to_string()));
}
