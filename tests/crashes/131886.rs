//@ known-bug: #131886
//@ compile-flags: -Zvalidate-mir --crate-type=lib
#![feature(trait_upcasting, type_alias_impl_trait)]

type Tait = impl Sized;

trait Foo<'a>: Bar<'a, 'a, Tait> {}
trait Bar<'a, 'b, T> {}

fn test_correct3<'a>(x: &dyn Foo<'a>, _: Tait) {
    let _ = x as &dyn Bar<'_, '_, ()>;
}
