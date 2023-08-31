#![feature(rustc_attrs, type_alias_impl_trait, impl_trait_in_assoc_type)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

type NotCapturedEarly<'a> = impl Sized; //~ [o]

type CapturedEarly<'a> = impl Sized + Captures<'a>; //~ [o]

type NotCapturedLate<'a> = dyn for<'b> Iterator<Item = impl Sized>; //~ [o]

type CapturedLate<'a> = dyn for<'b> Iterator<Item = impl Sized + Captures<'b>>; //~ [o]

type Captured<'a> = dyn for<'b> Iterator<Item = impl Sized + Captures<'a> + Captures<'b>>; //~ [o]

type Bar<'a, 'b: 'b, T> = impl Sized; //~ ERROR [o, o, o]

trait Foo<'i> {
    type ImplicitCapturedEarly<'a>;

    type ExplicitCaptureEarly<'a>;

    type ImplicitCaptureLate<'a>;

    type ExplicitCaptureLate<'a>;
}

impl<'i> Foo<'i> for &'i () {
    type ImplicitCapturedEarly<'a> = impl Sized; //~ [o, o]

    type ExplicitCaptureEarly<'a> = impl Sized + Captures<'i>; //~ [o, o]

    type ImplicitCaptureLate<'a> = impl Sized; //~ [o, o]

    type ExplicitCaptureLate<'a> = impl Sized + Captures<'a>; //~ [o, o]
}

impl<'i> Foo<'i> for () {
    type ImplicitCapturedEarly<'a> = impl Sized; //~ [o, o]

    type ExplicitCaptureEarly<'a> = impl Sized + Captures<'i>; //~ [o, o]

    type ImplicitCaptureLate<'a> = impl Sized; //~ [o, o]

    type ExplicitCaptureLate<'a> = impl Sized + Captures<'a>; //~ [o, o]
}

fn main() {}
