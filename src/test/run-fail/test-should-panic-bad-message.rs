// compile-flags: --test

// error-pattern:panicked at 'bar'
// check-stdout
#[test]
#[should_panic(expected = "foo")]
pub fn test_bar() {
    panic!("bar")
}
