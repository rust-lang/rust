// no-prefer-dynamic
// compile-flags: --test -Cpanic=abort -Zpanic_abort_tests
// run-flags: --test-threads=1 --nocapture
// run-fail
// check-run-results
// exec-env:RUST_BACKTRACE=0
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

// ignore-wasm no panic or subprocess support
// ignore-emscripten no panic or subprocess support
// ignore-sgx no subprocess support

#![cfg(test)]

use std::io::Write;

#[test]
fn it_works() {
    println!("about to succeed");
    assert_eq!(1 + 1, 2);
}

#[test]
#[should_panic]
fn it_panics() {
    println!("about to panic");
    assert_eq!(1 + 1, 4);
}

#[test]
fn it_fails() {
    println!("about to fail");
    assert_eq!(1 + 1, 4);
}

#[test]
fn it_writes_to_stdio() {
    println!("hello, world");
    writeln!(std::io::stdout(), "testing123").unwrap();
    writeln!(std::io::stderr(), "testing321").unwrap();
}
