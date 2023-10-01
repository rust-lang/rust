#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn not_captured_early<'a: 'a>() -> impl Sized {} //~ [*]

fn captured_early<'a: 'a>() -> impl Sized + Captures<'a> {} //~ [*, o]

fn not_captured_late<'a>(_: &'a ()) -> impl Sized {} //~ []

fn captured_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {} //~ [o]

fn main() {}
