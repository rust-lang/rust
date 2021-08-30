struct Foo<const NAME: &'static str>; //~ ERROR `&'static str` is forbidden
fn main() {}
