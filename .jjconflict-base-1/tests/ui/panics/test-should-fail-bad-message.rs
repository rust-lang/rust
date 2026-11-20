//@ run-fail
//@ check-stdout
//@ compile-flags: --test
//@ needs-unwind

#[test]
#[should_panic(expected = "foobar")]
fn test_foo() {
    panic!("blah")
}
