struct Foo;

impl Foo {
    fn foo(self: *const Self) {}
    //~^ ERROR raw pointer `self` is unstable
}

trait Bar {
    fn bar(self: *const Self);
    //~^ ERROR raw pointer `self` is unstable
}

impl Bar for () {
    fn bar(self: *const Self) {}
    //~^ ERROR raw pointer `self` is unstable
}

fn main() {}
