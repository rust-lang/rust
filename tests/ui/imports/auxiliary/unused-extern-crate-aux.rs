#![deny(dead_code)]

mod inner {
    pub trait Trait {
        fn f(&self) { f(); }
    }

    impl Trait for isize {}

    fn f() {}
}

pub fn foo() {
    let a = &1isize as &dyn inner::Trait;
    a.f();
}
