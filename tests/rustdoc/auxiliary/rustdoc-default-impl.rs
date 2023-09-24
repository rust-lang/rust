#![feature(rustc_attrs)]

pub mod bar {
    use std::marker;

    #[rustc_auto_trait]
    pub trait Bar {}

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
