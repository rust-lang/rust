// run-fail
//@compile-flags: --test
// check-stdout
//@ignore-target-emscripten no processes

#[test]
#[should_panic(expected = "foo")]
pub fn test_bar() {
    panic!("bar")
}
