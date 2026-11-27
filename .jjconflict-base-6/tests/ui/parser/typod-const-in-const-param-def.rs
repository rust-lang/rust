pub fn foo<Const N: u8>() {}
//~^ ERROR `const` keyword was mistyped as `Const`

pub fn bar<Const>() {}
// OK

pub fn baz<Const N: u8, T>() {}
//~^ ERROR `const` keyword was mistyped as `Const`

pub fn qux<T, Const N: u8>() {}
//~^ ERROR `const` keyword was mistyped as `Const`

pub fn quux<T, Const N: u8, U>() {}
//~^ ERROR `const` keyword was mistyped as `Const`

fn main() {}
