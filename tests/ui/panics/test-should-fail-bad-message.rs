// run-fail
// check-stdout
// compile-flags: --test
// ignore-emscripten

#[test]
#[should_panic(expected = "foobar")]
fn test_foo() {
    panic!("blah")
}
