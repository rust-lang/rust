// run-fail
// compile-flags: --test
// check-stdout

#[test]
#[should_panic(expected = "foo")]
pub fn test_explicit() {
    panic!()
}
