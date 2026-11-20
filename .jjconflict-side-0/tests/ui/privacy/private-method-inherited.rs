// Tests that inherited visibility applies to methods.

mod a {
    pub struct Foo;

    impl Foo {
        fn f(self) {}
    }
}

fn main() {
    let x = a::Foo;
    x.f();  //~ ERROR method `f` is private
}
