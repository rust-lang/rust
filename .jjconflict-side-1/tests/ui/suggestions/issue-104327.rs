trait Bar {}

trait Foo {
    fn f() {}
}

impl Foo for dyn Bar {}

fn main() {
    Foo::f();
    //~^ ERROR cannot call associated function on trait without specifying the corresponding `impl` type
}
