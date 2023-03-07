// no-prefer-dynamic
// compile-flags: --test -Cpanic=abort -Zpanic_abort_tests
// run-flags: --test-threads=1
// run-fail
// check-run-results
// exec-env:RUST_BACKTRACE=0
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

// ignore-wasm no panic or subprocess support
// ignore-emscripten no panic or subprocess support
// ignore-sgx no subprocess support

#![cfg(test)]

use std::io::Write;
use std::env;

#[test]
fn it_works() {
    assert_eq!(1 + 1, 2);
}

#[test]
#[should_panic]
fn it_panics() {
    assert_eq!(1 + 1, 4);
}

#[test]
fn it_fails() {
    println!("hello, world");
    writeln!(std::io::stdout(), "testing123").unwrap();
    writeln!(std::io::stderr(), "testing321").unwrap();
    assert_eq!(1 + 1, 5);
}

#[test]
fn it_exits() {
    std::process::exit(123);
}

#[test]
fn no_residual_environment() {
    for (key, _) in env::vars() {
        // Look for keys like __RUST_TEST_INVOKE.
        if key.contains("TEST_INVOKE") {
            panic!("shouldn't have '{}' in environment", key);
        }
    }
}
