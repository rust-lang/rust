//@ check-pass

#![feature(type_alias_impl_trait)]

mod foo {
    pub trait Foo {
        // This was reachable in https://github.com/rust-lang/rust/issues/100800
        fn foo(&self) {
            unreachable!()
        }
    }
    impl<T> Foo for T {}

    pub struct B;
    impl B {
        fn foo(&self) {}
    }
    pub type Input = impl Foo;
    fn bop() -> Input {
        super::run1(|x: B| x.foo(), B);
        super::run2(|x: B| x.foo(), B);
        panic!()
    }
}
use foo::*;

fn run1<F: FnOnce(Input)>(f: F, i: Input) {
    f(i)
}
fn run2<F: FnOnce(B)>(f: F, i: B) {
    f(i)
}

fn main() {}
