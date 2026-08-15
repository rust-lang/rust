//@ known-bug: #152295
#![feature(type_alias_impl_trait)]
type Tait = impl Sized;
trait Foo: Bar<Tait> {}
trait Bar<T> {
    fn bar(&self);
}
fn test_correct2(x: &dyn Foo) {
    x.bar();
}
fn main() {}
