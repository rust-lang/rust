trait Foo {}

impl<T: Fn(&())> Foo for T {}

fn baz<T: Foo>(_: T) {}

fn main() {
    baz(|_| ());
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR mismatched types
    baz(|x| ());
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR mismatched types
}
