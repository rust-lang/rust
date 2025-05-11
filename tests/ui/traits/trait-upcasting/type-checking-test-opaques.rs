#![feature(type_alias_impl_trait)]

//@ check-pass

type Tait = impl Sized;

trait Foo<'a>: Bar<'a, 'a, Tait> {}
trait Bar<'a, 'b, T> {}

fn test_correct(x: &dyn Foo<'static>) {
    let _ = x as &dyn Bar<'static, 'static, Tait>;
}

fn test_correct2<'a>(x: &dyn Foo<'a>) {
    let _ = x as &dyn Bar<'_, '_, Tait>;
}

#[define_opaque(Tait)]
fn test_correct3<'a>(x: &dyn Foo<'a>, _: Tait) {
    let _ = x as &dyn Bar<'_, '_, ()>;
}

fn main() {}
