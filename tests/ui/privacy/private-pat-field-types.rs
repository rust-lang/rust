mod foo {
    struct Bar;
    pub enum Foo {
        #[allow(private_interfaces)]
        A(Bar),
    }
}
fn foo_bar(v: foo::Foo) {
    match v {
        foo::Foo::A(_) => {} //~ ERROR type `Bar` is private
    }
}

fn main() {}
