// check-stdout
// error-pattern:thread 'test_foo' panicked at
// compile-flags: --test
// ignore-emscripten

#[test]
fn test_foo() {
    panic!()
}
