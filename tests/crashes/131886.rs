//@ known-bug: #131886
//@ compile-flags: -Zvalidate-mir
#![feature(type_alias_impl_trait)]

type Tait = impl Sized;

trait Foo<'a>: Bar<'a, 'a, Tait> {}
trait Bar<'a, 'b, T> {}

#[define_opaque(Tait)]
fn test_correct3<'a>(x: &dyn Foo<'a>, _: Tait) {
    let _ = x as &dyn Bar<'_, '_, ()>;
}

fn main() {}
