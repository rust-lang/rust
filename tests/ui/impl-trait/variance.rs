//@ revisions: old new e2024
//@[e2024] edition: 2024
//@[e2024] compile-flags: -Z unstable-options

#![cfg_attr(new, feature(lifetime_capture_rules_2024))]

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn not_captured_early<'a: 'a>() -> impl Sized {}
//[old]~^ [*]
//[new]~^^ [*, o]
//[e2024]~^^^ [*, o]

fn captured_early<'a: 'a>() -> impl Sized + Captures<'a> {} //~ [*, o]

fn not_captured_late<'a>(_: &'a ()) -> impl Sized {}
//[old]~^ []
//[new]~^^ [o]
//[e2024]~^^^ [o]

fn captured_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {} //~ [o]

fn main() {}
