#![feature(rustc_attrs, type_alias_impl_trait, impl_trait_in_assoc_type)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

type NotCapturedEarly<'a> = impl Sized; //~ [o]

type CapturedEarly<'a> = impl Sized + Captures<'a>; //~ [o]

// TAIT does *not* capture `'b`
type NotCapturedLate<'a> = dyn for<'b> Iterator<Item = impl Sized>; //~ [o]

// TAIT does *not* capture `'b`
type Captured<'a> = dyn for<'b> Iterator<Item = impl Sized + Captures<'a>>; //~ [o]

type Bar<'a, 'b: 'b, T> = impl Sized; //~ ERROR [o, o, o]

trait Foo<'i> {
    type ImplicitCapture<'a>;

    type ExplicitCaptureFromHeader<'a>;

    type ExplicitCaptureFromGat<'a>;
}

impl<'i> Foo<'i> for &'i () {
    type ImplicitCapture<'a> = impl Sized; //~ [o, o]

    type ExplicitCaptureFromHeader<'a> = impl Sized + Captures<'i>; //~ [o, o]

    type ExplicitCaptureFromGat<'a> = impl Sized + Captures<'a>; //~ [o, o]
}

impl<'i> Foo<'i> for () {
    type ImplicitCapture<'a> = impl Sized; //~ [o, o]

    type ExplicitCaptureFromHeader<'a> = impl Sized + Captures<'i>; //~ [o, o]

    type ExplicitCaptureFromGat<'a> = impl Sized + Captures<'a>; //~ [o, o]
}

fn main() {}
