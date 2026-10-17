//@ known-bug: #148094
//@ compile-flags: -Zvalidate-mir
#![feature(type_alias_impl_trait)]
type Tait = impl Sized;
trait Foo: Bar<Tait> {}
trait Bar<T> {}
#[define_opaque(Tait)]
fn test_correct3(x: &dyn Foo) -> &dyn Bar<()> {
    x
}
fn main() {}
