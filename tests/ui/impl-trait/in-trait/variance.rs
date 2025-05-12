#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Foo<'i> {
    fn implicit_capture_early<'a: 'a>() -> impl Sized {}
    //~^ ERROR [Self: o, 'i: o, 'a: *, 'i: o, 'a: o]

    fn explicit_capture_early<'a: 'a>() -> impl Sized + use<'i, 'a, Self> {}
    //~^ ERROR [Self: o, 'i: o, 'a: *, 'i: o, 'a: o]

    fn not_captured_early<'a: 'a>() -> impl Sized + use<'i, Self> {}
    //~^ ERROR [Self: o, 'i: o, 'a: *, 'i: o]

    fn implicit_capture_late<'a>(_: &'a ()) -> impl Sized {}
    //~^ ERROR [Self: o, 'i: o, 'i: o, 'a: o]

    fn explicit_capture_late<'a>(_: &'a ()) -> impl Sized + use<'i, 'a, Self> {}
    //~^ ERROR [Self: o, 'i: o, 'i: o, 'a: o]

    fn not_captured_late<'a>(_: &'a ()) -> impl Sized + use<'i, Self> {}
    //~^ ERROR [Self: o, 'i: o, 'i: o]
}

fn main() {}
