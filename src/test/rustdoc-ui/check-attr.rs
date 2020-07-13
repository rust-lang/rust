#![deny(invalid_codeblock_attributes)]

/// foo
//~^ ERROR
//~^^ ERROR
//~^^^ ERROR
///
/// ```compile-fail,compilefail,comPile_fail
/// boo
/// ```
pub fn foo() {}

/// bar
//~^ ERROR
//~^^ ERROR
//~^^^ ERROR
///
/// ```should-panic,shouldpanic,sHould_panic
/// boo
/// ```
pub fn bar() {}

/// foobar
//~^ ERROR
//~^^ ERROR
//~^^^ ERROR
///
/// ```no-run,norun,no_Run
/// boo
/// ```
pub fn foobar() {}

/// barfoo
//~^ ERROR
//~^^ ERROR
//~^^^ ERROR
///
/// ```allow-fail,allowfail,alLow_fail
/// boo
/// ```
pub fn barfoo() {}

/// b
//~^ ERROR
//~^^ ERROR
//~^^^ ERROR
///
/// ```test-harness,testharness,teSt_harness
/// boo
/// ```
pub fn b() {}
