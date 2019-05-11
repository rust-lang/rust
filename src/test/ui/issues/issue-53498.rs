pub mod test {
    pub struct A;
    pub struct B;
    pub struct Foo<T>(T);

    impl Foo<A> {
        fn foo() {}
    }

    impl Foo<B> {
        fn foo() {}
    }
}

fn main() {
    test::Foo::<test::B>::foo(); //~ ERROR method `foo` is private
}
