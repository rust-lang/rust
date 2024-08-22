#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

trait Foo<'i> {
    fn implicit_capture_early<'a: 'a>() -> impl Sized {}
    //~^ [Self: o, 'i: *, 'a: *, 'a: o, 'i: o]

    fn explicit_capture_early<'a: 'a>() -> impl Sized + Captures<'a> {}
    //~^ [Self: o, 'i: *, 'a: *, 'a: o, 'i: o]

    fn implicit_capture_late<'a>(_: &'a ()) -> impl Sized {}
    //~^ [Self: o, 'i: *, 'a: o, 'i: o]

    fn explicit_capture_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {}
    //~^ [Self: o, 'i: *, 'a: o, 'i: o]
}

fn main() {}
