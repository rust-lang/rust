#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn no_elided_lt() -> impl use<'_> Sized {}
//~^ ERROR missing lifetime specifier
//~| ERROR expected lifetime parameter in `use<...>` precise captures list, found `'_`

fn static_lt() -> impl use<'static> Sized {}
//~^ ERROR expected lifetime parameter in `use<...>` precise captures list, found `'static`

fn missing_lt() -> impl use<'missing> Sized {}
//~^ ERROR use of undeclared lifetime name `'missing`

fn main() {}
