//@ revisions: same_name different_name

pub mod test {
    pub struct A;
    pub struct B;
    pub struct Foo<T>(T);

    impl Foo<A> {
        fn foo() {}
    }

    impl Foo<B> {
        #[cfg(same_name)]
        fn foo() {}
        #[cfg(different_name)]
        fn bar() {}
    }
}

fn main() {
    test::Foo::<test::B>::foo();
    //[same_name]~^ ERROR associated function `foo` is private
    //[different_name]~^^ ERROR no function or associated item named `foo` found for struct `Foo<B>`
}
