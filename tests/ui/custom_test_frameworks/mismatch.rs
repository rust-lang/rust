//@ aux-build:example_runner.rs
//@ compile-flags:--test
#![feature(custom_test_frameworks)]
#![test_runner(example_runner::runner)]

extern crate example_runner;

#[test]
fn wrong_kind(){}
//~^ ERROR trait bound `TestDescAndFn: Testable` is not satisfied
