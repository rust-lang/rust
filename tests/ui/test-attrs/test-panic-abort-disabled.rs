// error-pattern:building tests with panic=abort is not supported
// no-prefer-dynamic
// compile-flags: --test -Cpanic=abort -Zpanic-abort-tests=no
// run-flags: --test-threads=1

// needs-unwind
// ignore-wasm no panic or subprocess support
// ignore-emscripten no panic or subprocess support

#![cfg(test)]

#[test]
fn it_works() {
    assert_eq!(1 + 1, 2);
}

#[test]
#[should_panic]
fn it_panics() {
    assert_eq!(1 + 1, 4);
}
