trait Foo {}

impl<T> Foo for Bar<T> {} //~ ERROR cannot find type `Bar` in this scope

fn main() {}
