mod a {
    struct Foo;
    impl Foo { pub fn new() {} }
    enum Bar {}
    impl Bar { pub fn new() {} }
}

fn main() {
    a::Foo::new();
    //~^ ERROR: struct `Foo` is private
    a::Bar::new();
    //~^ ERROR: enum `Bar` is private
}
