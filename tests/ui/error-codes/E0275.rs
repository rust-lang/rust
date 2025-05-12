trait Foo {}

struct Bar<T>(T);

impl<T> Foo for T where Bar<T>: Foo {} //~ ERROR E0275

fn main() {
}
