#![feature(auto_traits)]

pub mod bar {
    use std::marker;

    pub auto trait Bar {}

    pub trait Foo {
        fn foo(&self) {}
    }

    impl Foo {
        pub fn test<T: Bar>(&self) {}
    }

    pub struct TypeId;

    impl TypeId {
        pub fn of<T: Bar + ?Sized>() -> TypeId {
            panic!()
        }
    }
}
