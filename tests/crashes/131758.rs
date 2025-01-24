//@ known-bug: #131758
#![feature(unboxed_closures)]
trait Foo {}

impl<T: Fn<(i32,)>> Foo for T {}

fn baz<T: Foo>(_: T) {}

fn main() {
    baz(|x| ());
}
