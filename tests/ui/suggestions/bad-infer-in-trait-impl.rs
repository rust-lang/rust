trait Foo {
    fn bar();
}

impl Foo for () {
    fn bar(s: _) {}
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
}

fn main() {}
