#![feature(rustc_attrs, type_alias_impl_trait, impl_trait_in_assoc_type)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

type NotCapturedEarly<'a> = impl Sized; //~ [*, o]
//~^ ERROR: unconstrained opaque type

type CapturedEarly<'a> = impl Sized + Captures<'a>; //~ [*, o]
//~^ ERROR: unconstrained opaque type

type NotCapturedLate<'a> = dyn for<'b> Iterator<Item = impl Sized>; //~ [*, o, o]
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
//~| ERROR: unconstrained opaque type

type Captured<'a> = dyn for<'b> Iterator<Item = impl Sized + Captures<'a>>; //~ [*, o, o]
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
//~| ERROR: unconstrained opaque type

type Bar<'a, 'b: 'b, T> = impl Sized; //~ ERROR [*, *, o, o, o]
//~^ ERROR: unconstrained opaque type

trait Foo<'i> {
    type ImplicitCapture<'a>;

    type ExplicitCaptureFromHeader<'a>;

    type ExplicitCaptureFromGat<'a>;
}

impl<'i> Foo<'i> for &'i () {
    type ImplicitCapture<'a> = impl Sized; //~ [*, *, o, o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromHeader<'a> = impl Sized + Captures<'i>; //~ [*, *, o, o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromGat<'a> = impl Sized + Captures<'a>; //~ [*, *, o, o]
    //~^ ERROR: unconstrained opaque type
}

impl<'i> Foo<'i> for () {
    type ImplicitCapture<'a> = impl Sized; //~ [*, *, o, o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromHeader<'a> = impl Sized + Captures<'i>; //~ [*, *, o, o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromGat<'a> = impl Sized + Captures<'a>; //~ [*, *, o, o]
    //~^ ERROR: unconstrained opaque type
}

fn main() {}
