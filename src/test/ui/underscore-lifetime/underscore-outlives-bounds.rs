// Regression test to check that `'b: '_` gets an error, because it's
// basically useless.
//
// #54902

trait Foo<'a> {}
impl<'b: '_> Foo<'b> for i32 {} //~ ERROR missing lifetime specifier [E0106]
fn main() { }
