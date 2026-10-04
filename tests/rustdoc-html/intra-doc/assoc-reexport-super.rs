// Regression test for #93205

#![crate_name = "foo"]

mod generated {
    pub struct MyNewType;
    impl MyNewType {
        pub const FOO: Self = Self;
    }
}

pub use generated::MyNewType;

mod prelude {
    impl super::MyNewType {
        /// An alias for [`Self::FOO`].
        //@ has 'foo/type.MyNewType.html' '//a[@href="type.MyNewType.html#associatedconstant.FOO"]' 'Self::FOO'
        pub const FOO2: Self = Self::FOO;
    }
}
