fn lifetime_in_bounds<'a>(x: &'a ()) -> impl Into<&'a ()> + use<> { x }
//~^ ERROR `impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list

fn lifetime_in_hidden<'a>(x: &'a ()) -> impl Sized + use<> { x }
//~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds

fn main() {}
