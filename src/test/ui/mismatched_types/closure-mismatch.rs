trait Foo {}

impl<T: Fn(&())> Foo for T {}

fn baz<T: Foo>(_: T) {}

fn main() {
    baz(|_| ()); //~ ERROR E0308
}
