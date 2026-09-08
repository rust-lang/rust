//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]
#![rustc_dump_variances_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn not_captured_early<'a: 'a>() -> impl Sized {}

fn captured_early<'a: 'a>() -> impl Sized + Captures<'a> {}

fn not_captured_late<'a>(_: &'a ()) -> impl Sized {}

fn captured_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {}
