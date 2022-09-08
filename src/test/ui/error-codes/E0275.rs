trait Foo {}

struct Bar<T>(T);

impl<T> Foo for T where Bar<T>: Foo {}

fn takes_foo<T: Foo>() {}

fn main() {
    takes_foo::<()>(); //~ ERROR E0275
}
