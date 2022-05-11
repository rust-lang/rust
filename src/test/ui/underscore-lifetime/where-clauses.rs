trait Foo<'a> {}

impl<'b: '_> Foo<'b> for i32 {} //~ ERROR missing lifetime specifier [E0106]

impl<T: '_> Foo<'static> for Vec<T> {} //~ ERROR missing lifetime specifier [E0106]

fn main() { }
