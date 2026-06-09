//@ run-fail
//@ check-stdout
//@ compile-flags: --test
//@ needs-subprocess

#[test]
fn test_foo() {
    panic!()
}
