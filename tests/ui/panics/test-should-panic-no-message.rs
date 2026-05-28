//@ run-fail
//@ compile-flags: --test
//@ check-stdout
//@ needs-subprocess

#[test]
#[should_panic(expected = "foo")]
pub fn test_explicit() {
    panic!()
}
