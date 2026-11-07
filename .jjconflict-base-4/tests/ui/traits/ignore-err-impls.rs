pub struct S;

trait Generic<T> {}

impl<'a, T> Generic<&'a T> for S {}
impl Generic<Type> for S {}
//~^ ERROR cannot find type `Type` in this scope

fn main() {}
