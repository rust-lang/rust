//! This test used to report that the method call cannot
//! call the private method `Foo<A>::foo`, even though the user
//! explicitly selected `Foo<B>::foo`. This is because we only
//! looked for methods of the right name, without properly checking
//! the `Self` type

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
