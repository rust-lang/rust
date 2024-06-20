trait Foo<T> {}

impl Foo<T: Default> for String {}
//~^ ERROR associated item constraints are not allowed here
//~| HELP declare the type parameter right after the `impl` keyword

impl Foo<T: 'a + Default> for u8 {}
//~^ ERROR associated item constraints are not allowed here
//~| HELP declare the type parameter right after the `impl` keyword
//~| ERROR use of undeclared lifetime name `'a`
//~| HELP consider introducing lifetime `'a` here

impl<T> Foo<T: Default> for u16 {}
//~^ ERROR associated item constraints are not allowed here
//~| HELP declare the type parameter right after the `impl` keyword

impl<'a> Foo<T: 'a + Default> for u32 {}
//~^ ERROR associated item constraints are not allowed here
//~| HELP declare the type parameter right after the `impl` keyword

trait Bar<T, K> {}

impl Bar<T: Default, K: Default> for String {}
//~^ ERROR associated item constraints are not allowed here
//~| HELP declare the type parameter right after the `impl` keyword

impl<T> Bar<T, K: Default> for u8 {}
//~^ ERROR trait takes 2 generic arguments but 1 generic argument was supplied
//~| HELP add missing generic argument
//~| ERROR associated item constraints are not allowed here
//~| HELP declare the type parameter right after the `impl` keyword

fn main() {}
