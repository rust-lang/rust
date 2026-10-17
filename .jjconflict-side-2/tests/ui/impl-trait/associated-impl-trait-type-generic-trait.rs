#![feature(impl_trait_in_assoc_type)]
//@ build-pass (FIXME(62277): could be check-pass?)

trait Bar {}
struct Dummy<U>(U);
impl<V> Bar for Dummy<V> {}

trait Foo<T> {
    type Assoc: Bar;
    fn foo(t: T) -> Self::Assoc;
}

impl<W> Foo<W> for i32 {
    type Assoc = impl Bar;
    fn foo(w: W) -> Self::Assoc {
        Dummy(w)
    }
}

struct NonGeneric;
impl Bar for NonGeneric {}

impl<W> Foo<W> for u32 {
    type Assoc = impl Bar;
    fn foo(_: W) -> Self::Assoc {
        NonGeneric
    }
}

fn main() {}
