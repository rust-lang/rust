//@ compile-flags:--test

#![deny(rustdoc::invalid_codeblock_attributes)]

//~vvv ERROR unknown attribute `compile-fail`
//~| ERROR unknown attribute `compilefail`
//~| ERROR unknown attribute `comPile_fail`
/// foo
///
/// ```compile-fail,compilefail,comPile_fail
/// boo
/// ```
pub fn foo() {}

//~vvv ERROR unknown attribute `should-panic`
//~| ERROR unknown attribute `shouldpanic`
//~| ERROR unknown attribute `shOuld_panic`
/// bar
///
/// ```should-panic,shouldpanic,shOuld_panic
/// boo
/// ```
pub fn bar() {}

//~vvv ERROR unknown attribute `no-run`
//~| ERROR unknown attribute `norun`
//~| ERROR unknown attribute `nO_run`
/// foobar
///
/// ```no-run,norun,nO_run
/// boo
/// ```
pub fn foobar() {}

//~vvv ERROR unknown attribute `test-harness`
//~| ERROR unknown attribute `testharness`
//~| ERROR unknown attribute `tesT_harness`
/// b
///
/// ```test-harness,testharness,tesT_harness
/// boo
/// ```
pub fn b() {}
