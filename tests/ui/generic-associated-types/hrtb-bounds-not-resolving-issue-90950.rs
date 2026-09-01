//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/90950>.
// Used to say bound not satisfied

trait Yokeable<'a>: 'static {
    type Output: 'a;
}


trait IsCovariant<'a> {}

struct Yoke<Y: for<'a> Yokeable<'a>> {
    data: Y,
}

fn upcast<Y>(x: Yoke<Y>) -> Yoke<Box<dyn IsCovariant<'static> + 'static>>
where
    Y: for<'a> Yokeable<'a>,
    for<'a> <Y as Yokeable<'a>>::Output: IsCovariant<'a>,
{
    unimplemented!()
}


impl<'a> Yokeable<'a> for Box<dyn IsCovariant<'static> + 'static> {
    type Output = Box<dyn IsCovariant<'a> + 'a>;
}

use std::borrow::*;
impl<'a, T: ToOwned + ?Sized> Yokeable<'a> for Cow<'static, T> {
    type Output = Cow<'a, T>;
}
impl<'a, T: ToOwned + ?Sized> IsCovariant<'a> for Cow<'a, T> {}


fn upcast_yoke(y: Yoke<Cow<'static, str>>) -> Yoke<Box<dyn IsCovariant<'static> + 'static>> {
    upcast(y)
}

fn main() {}
