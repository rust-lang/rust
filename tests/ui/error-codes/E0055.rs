#![recursion_limit="4"]
struct Foo;

impl Foo {
    fn foo(&self) {}
}

fn main() {
    let foo = Foo;
    let ref_foo = &&&&&Foo;
    ref_foo.foo();
    //~^ ERROR E0055
}
