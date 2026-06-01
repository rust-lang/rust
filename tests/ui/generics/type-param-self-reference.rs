pub struct Foo<Bar=Bar>(Bar); //~ ERROR E0128
pub struct Baz(Foo);
fn main() {}
