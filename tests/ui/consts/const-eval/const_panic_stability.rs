//@ revisions: e2018 e2021
//@[e2018] edition:2018
//@[e2021] edition:2021
//@[e2018] check-pass
#![crate_type = "lib"]
#![stable(feature = "foo", since = "1.0.0")]
#![feature(staged_api)]

#[stable(feature = "foo", since = "1.0.0")]
#[rustc_const_stable(feature = "foo", since = "1.0.0")]
const fn foo() {
    assert!(false);
    assert!(false, "foo");
    panic!({ "foo" });
    //[e2018]~^ WARNING panic message is not a string literal
    //[e2021]~^^ ERROR format argument must be a string literal
}
