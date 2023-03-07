use inner::Trait;

mod inner {
    pub struct Foo;
    pub trait Trait {
        fn f(&self);
        fn f_ufcs(&self);
    }

    impl Trait for Foo {
        fn f(&self) { }
        fn f_ufcs(&self) { }
    }
}

pub trait Outer {
    fn foo<T: Trait>(&self, t: T) { t.f(); }
    fn foo_ufcs<T: Trait>(&self, t: T) { T::f(&t); }
}

impl Outer for isize {}

pub fn foo<T: Outer>(t: T) {
    t.foo(inner::Foo);
}
pub fn foo_ufcs<T: Outer>(t: T) {
    T::foo_ufcs(&t, inner::Foo)
}
