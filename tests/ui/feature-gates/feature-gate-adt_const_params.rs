struct Foo<const NAME: &'static [u8]>; //~ ERROR `&'static [u8]` is forbidden
fn main() {}
