//@ revisions: old e2024
//@[e2024] edition: 2024

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn not_captured_early<'a: 'a>() -> impl Sized {}
//[old]~^ ERROR ['a: *]
//[e2024]~^^ ERROR ['a: *, 'a: o]

fn captured_early<'a: 'a>() -> impl Sized + Captures<'a> {} //~ ERROR ['a: *, 'a: o]

fn not_captured_late<'a>(_: &'a ()) -> impl Sized {}
//[old]~^ ERROR []
//[e2024]~^^ ERROR ['a: o]

fn captured_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {} //~ ERROR ['a: o]

fn main() {}
