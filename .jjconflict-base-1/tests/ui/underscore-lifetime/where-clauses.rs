trait Foo<'a> {}

impl<'b: '_> Foo<'b> for i32 {} //~ ERROR `'_` cannot be used here

impl<T: '_> Foo<'static> for Vec<T> {} //~ ERROR `'_` cannot be used here

fn main() { }
