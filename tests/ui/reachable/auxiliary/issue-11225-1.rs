mod inner {
    pub trait Trait {
        fn f(&self) { f(); }
        fn f_ufcs(&self) { f_ufcs(); }
    }

    impl Trait for isize {}

    fn f() {}
    fn f_ufcs() {}
}

pub fn foo<T: inner::Trait>(t: T) {
    t.f();
}
pub fn foo_ufcs<T: inner::Trait>(t: T) {
    T::f_ufcs(&t);
}
