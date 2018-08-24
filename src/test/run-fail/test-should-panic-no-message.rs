// compile-flags: --test

// error-pattern:panicked at 'explicit panic'
// check-stdout
#[test]
#[should_panic(expected = "foo")]
pub fn test_explicit() {
    panic!()
}
