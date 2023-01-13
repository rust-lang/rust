// Regression test for issue #57807 - ensure
// that we properly unify associated types within
// a type alias impl trait
// check-pass
#![feature(type_alias_impl_trait)]

trait Bar {
    type A;
}

impl Bar for () {
    type A = ();
}

trait Foo {
    type A;
    type B: Bar<A = Self::A>;

    fn foo() -> Self::B;
}

impl Foo for () {
    type A = ();
    type B = impl Bar<A = Self::A>;

    fn foo() -> Self::B {
        ()
    }
}

fn main() {}
