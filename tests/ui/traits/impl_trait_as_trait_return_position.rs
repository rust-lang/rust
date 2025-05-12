//@ check-pass

trait A {
    type Foo;
}

impl<T> A for T {
    type Foo = ();
}

fn foo() -> impl std::borrow::Borrow<<u8 as A>::Foo> {
    ()
}

fn main() {
    foo();
}
