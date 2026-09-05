//@ edition: 2015
//@ compile-flags: --crate-type=lib
//@ revisions: pass fail
//@[pass] check-pass
#![cfg(fail)]

type A0 = dyn;
//[fail]~^ ERROR cannot find type `dyn` in this scope

type A1 = dyn::dyn;
//[fail]~^ ERROR cannot find module or crate `dyn` in this scope

type A2 = dyn<dyn, dyn>;
//[fail]~^ ERROR cannot find type `dyn` in this scope
//[fail]~| ERROR cannot find type `dyn` in this scope
//[fail]~| ERROR cannot find type `dyn` in this scope

type A3 = dyn<<dyn as dyn>::dyn>;
//[fail]~^ ERROR cannot find type `dyn` in this scope
//[fail]~| ERROR cannot find type `dyn` in this scope
//[fail]~| ERROR cannot find trait `dyn` in this scope

// issue: <https://github.com/rust-lang/rust/issues/157565>
type A4 = dyn + dyn;
//[fail]~^ ERROR cannot find trait `dyn` in this scope
//[fail]~| ERROR cannot find trait `dyn` in this scope
//[fail]~| WARN trait objects without an explicit `dyn` are deprecated
//[fail]~| WARN this is accepted in the current edition

// The `for<…> dyn …` -> `dyn for<…> …` recovery code used to incorrectly treat `dyn` as a keyword
// in Rust 2015 even it's not followed by a token in the "trigger set".
type A5 = for<> dyn;
//[fail]~^ ERROR cannot find trait `dyn` in this scope
//[fail]~| WARN trait objects without an explicit `dyn` are deprecated
//[fail]~| WARN this is accepted in the current edition
