struct Foo;

impl Foo {
    fn foo(self: *const Self) {}
    //~^ ERROR `*const Foo` cannot be used as the type of `self` without
}

trait Bar {
    fn bar(self: *const Self);
    //~^ ERROR `*const Self` cannot be used as the type of `self` without
}

impl Bar for () {
    fn bar(self: *const Self) {}
    //~^ ERROR `*const ()` cannot be used as the type of `self` without
}

fn main() {}
