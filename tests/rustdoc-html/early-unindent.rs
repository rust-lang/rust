// This is a regression for https://github.com/rust-lang/rust/issues/96079.

#![crate_name = "foo"]

pub mod app {
    pub struct S;

    impl S {
        //@ has 'foo/app/struct.S.html'
        //@ has - '//a[@href="../enums/enum.Foo.html#method.by_name"]' 'Foo::by_name'
        /**
        Doc comment hello! [`Foo::by_name`](`crate::enums::Foo::by_name`).
        */
        pub fn whatever(&self) {}
    }
}

pub mod enums {
    pub enum Foo {
        Bar,
    }

    impl Foo {
        pub fn by_name(&self) {}
    }
}
