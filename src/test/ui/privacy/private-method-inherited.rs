// Tests that inherited visibility applies to methods.

mod a {
    pub struct Foo;

    impl Foo {
        fn f(self) {}
    }
}

fn main() {
    let x = a::Foo;
    x.f();  //~ ERROR associated function `f` is private
}
