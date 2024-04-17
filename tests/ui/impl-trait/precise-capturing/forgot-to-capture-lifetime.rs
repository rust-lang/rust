#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn lifetime_in_bounds<'a>(x: &'a ()) -> impl use<> Into<&'a ()> { x }
//~^ ERROR `impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list

fn lifetime_in_hidden<'a>(x: &'a ()) -> impl use<> Sized { x }
//~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds

fn main() {}
