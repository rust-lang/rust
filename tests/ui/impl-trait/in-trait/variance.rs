#![feature(rustc_attrs, precise_capturing_in_traits)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Foo<'i> {
    fn implicit_capture_early<'a: 'a>() -> impl Sized {}
    //~^ [Self: o, 'i: o, 'a: *, 'a: o, 'i: o]

    fn explicit_capture_early<'a: 'a>() -> impl Sized + use<'i, 'a, Self> {}
    //~^ [Self: o, 'i: o, 'a: *, 'i: o, 'a: o]

    fn not_captured_early<'a: 'a>() -> impl Sized + use<'i, Self> {}
    //~^ [Self: o, 'i: o, 'a: *, 'i: o]

    fn implicit_capture_late<'a>(_: &'a ()) -> impl Sized {}
    //~^ [Self: o, 'i: o, 'a: o, 'i: o]

    fn explicit_capture_late<'a>(_: &'a ()) -> impl Sized + use<'i, 'a, Self> {}
    //~^ [Self: o, 'i: o, 'i: o, 'a: o]

    fn not_cpatured_late<'a>(_: &'a ()) -> impl Sized + use<'i, Self> {}
    //~^ [Self: o, 'i: o, 'i: o]
}

fn main() {}
