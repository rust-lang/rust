#![feature(object_safe_for_dispatch)]

trait Foo {
    fn f() {}
}

impl Foo for dyn Sized {}

fn main() {
    Foo::f();
    //~^ ERROR cannot call associated function on trait without specifying the corresponding `impl` type
}
