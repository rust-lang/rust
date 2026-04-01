trait Foo {
    fn bar();
}

impl Foo for () {
    fn bar(s: _) {}
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated functions
    //~| ERROR has 1 parameter but the declaration in trait `Foo::bar` has 0
}

fn main() {}
