#![feature(rustc_attrs, type_alias_impl_trait, impl_trait_in_assoc_type)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

type NotCapturedEarly<'a> = impl Sized; //~ ERROR ['a: *, 'a: o]
//~^ ERROR: unconstrained opaque type

type CapturedEarly<'a> = impl Sized + Captures<'a>; //~ ERROR ['a: *, 'a: o]
//~^ ERROR: unconstrained opaque type

type NotCapturedLate<'a> = dyn for<'b> Iterator<Item = impl Sized>; //~ ERROR ['a: *, 'a: o, 'b: o]
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
//~| ERROR: unconstrained opaque type

type Captured<'a> = dyn for<'b> Iterator<Item = impl Sized + Captures<'a>>; //~ ERROR ['a: *, 'a: o, 'b: o]
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
//~| ERROR: unconstrained opaque type

type Bar<'a, 'b: 'b, T> = impl Sized; //~ ERROR ['a: *, 'b: *, T: o, 'a: o, 'b: o]
//~^ ERROR: unconstrained opaque type

trait Foo<'i> {
    type ImplicitCapture<'a>;

    type ExplicitCaptureFromHeader<'a>;

    type ExplicitCaptureFromGat<'a>;
}

impl<'i> Foo<'i> for &'i () {
    type ImplicitCapture<'a> = impl Sized; //~ ERROR ['i: *, 'a: *, 'i: o, 'a: o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromHeader<'a> = impl Sized + Captures<'i>; //~ ERROR ['i: *, 'a: *, 'i: o, 'a: o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromGat<'a> = impl Sized + Captures<'a>; //~ ERROR ['i: *, 'a: *, 'i: o, 'a: o]
    //~^ ERROR: unconstrained opaque type
}

impl<'i> Foo<'i> for () {
    type ImplicitCapture<'a> = impl Sized; //~ ERROR ['i: *, 'a: *, 'i: o, 'a: o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromHeader<'a> = impl Sized + Captures<'i>; //~ ERROR ['i: *, 'a: *, 'i: o, 'a: o]
    //~^ ERROR: unconstrained opaque type

    type ExplicitCaptureFromGat<'a> = impl Sized + Captures<'a>; //~ ERROR ['i: *, 'a: *, 'i: o, 'a: o]
    //~^ ERROR: unconstrained opaque type
}

trait Nesting<'a> {
    type Output;
}
impl<'a> Nesting<'a> for &'a () {
    type Output = &'a ();
}
type NestedDeeply<'a> =
    impl Nesting< //~ ERROR ['a: *, 'a: o]
        'a,
        Output = impl Nesting< //~ ERROR ['a: *, 'a: o]
            'a,
            Output = impl Nesting< //~ ERROR ['a: *, 'a: o]
                'a,
                Output = impl Nesting< //~ ERROR ['a: *, 'a: o]
                    'a,
                    Output = impl Nesting<'a> //~ ERROR ['a: *, 'a: o]
                >
            >,
        >,
    >;
#[define_opaque(NestedDeeply)]
fn test<'a>() -> NestedDeeply<'a> {
    &()
}

fn main() {}
