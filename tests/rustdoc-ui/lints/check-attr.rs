#![deny(rustdoc::invalid_codeblock_attributes)]

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

/// b
//~^ ERROR
//~^^ ERROR
//~^^^ ERROR
///
/// ```test-harness,testharness,teSt_harness
/// boo
/// ```
pub fn b() {}

/// b
//~^ ERROR
///
/// ```rust2018
/// boo
/// ```
pub fn c() {}

/// b
//~^ ERROR
//~| ERROR
///
/// ```rust2018 shouldpanic
/// boo
/// ```
pub fn d() {}
