//! Test the `dyn` suggestion for bare trait objects whose first bound is a global path, like
//! `Box<::std::any::Any + 'static>`. In edition 2015 only that first bound gets parentheses,
//! since `dyn ::std::any::Any` would be parsed as the path `dyn::std::any::Any`. Later
//! editions need no parentheses. `unused_parens` is denied so that the fixed code also
//! fails if the suggestion adds parentheses that aren't needed.
//! Regression test for <https://github.com/rust-lang/rust/issues/63330>.

//@ revisions: e2015 e2018 e2021
//@[e2015] edition: 2015
//@[e2018] edition: 2018
//@[e2021] edition: 2021
//@ run-rustfix

#![crate_type = "lib"]
#![deny(bare_trait_objects)]
#![deny(unused_parens)]

pub fn issue_example(_: Box<::std::any::Any + 'static>) {}
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition

pub type AutoTrait = Box<::std::any::Any + Send>;
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition

pub type SingleBound = Box<::std::any::Any>;
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition

pub type InRef<'a> = &'a (::std::any::Any + 'static);
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition

pub type GlobalAutoTrait = Box<::std::any::Any + ::std::marker::Send + 'static>;
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition

pub type LifetimeFirst = Box<'static + ::std::any::Any>;
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition

pub type Binder = Box<for<'a> ::std::ops::Fn(&'a u8) + Send>;
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition

pub type FnSugar = Box<::std::ops::Fn(u8) -> u8 + Send>;
//[e2021]~^ ERROR expected a type, found a trait
//[e2015,e2018]~^^ ERROR trait objects without an explicit `dyn` are deprecated
//[e2015,e2018]~| WARN this is accepted in the current edition
