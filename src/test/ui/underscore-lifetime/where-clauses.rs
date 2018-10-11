trait Foo<'a> {}
impl<'b: '_> Foo<'b> for i32 {}
fn main() { }
