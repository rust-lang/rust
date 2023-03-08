// revisions: not_static yes_static
//[yes_static] check-pass

#[derive(Clone)]
struct Foo<'lt>(&'lt ());

impl Copy for Foo<'static> {}

#[derive(Clone)]
struct Bar<'lt>(Foo<'lt>);

#[cfg(not_static)]
impl<'any> Copy for Bar<'any> {}
//[not_static]~^ the trait `Copy` cannot be implemented for this type

#[cfg(yes_static)]
impl<'any> Copy for Bar<'static> {}

fn main() {}
