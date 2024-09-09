fn no_elided_lt() -> impl Sized + use<'_> {}
//~^ ERROR missing lifetime specifier
//~| ERROR expected lifetime parameter in `use<...>` precise captures list, found `'_`

fn static_lt() -> impl Sized + use<'static> {}
//~^ ERROR expected lifetime parameter in `use<...>` precise captures list, found `'static`

fn missing_lt() -> impl Sized + use<'missing> {}
//~^ ERROR use of undeclared lifetime name `'missing`

fn main() {}
