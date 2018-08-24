// check-stdout
// error-pattern:thread 'test_foo' panicked at
// compile-flags: --test
// ignore-emscripten

#[test]
#[should_panic(expected = "foobar")]
fn test_foo() {
    panic!("blah")
}
