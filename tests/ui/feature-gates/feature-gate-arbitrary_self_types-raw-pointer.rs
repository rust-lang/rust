struct Foo;

impl Foo {
    fn foo(self: *const Self) {}
    //~^ ERROR invalid `self` parameter type: `*const Foo`
}

trait Bar {
    fn bar(self: *const Self);
    //~^ ERROR invalid `self` parameter type: `*const Self`
}

impl Bar for () {
    fn bar(self: *const Self) {}
    //~^ ERROR invalid `self` parameter type: `*const ()`
}

fn main() {}
