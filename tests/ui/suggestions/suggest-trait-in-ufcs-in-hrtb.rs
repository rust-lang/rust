pub struct Bar<S>(S);

pub trait Foo {}

impl<S> Foo for Bar<S> where for<'a> <&'a S>::Item: Foo {}
//~^ ERROR ambiguous associated type

fn main() {}
