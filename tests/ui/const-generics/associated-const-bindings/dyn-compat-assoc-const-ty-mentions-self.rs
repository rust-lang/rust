// Ensure that we consider traits dyn *in*compatible if the type of any (type) assoc const
// mentions `Self` (barring "`Self` projections")

//@ dont-require-annotations: NOTE

#![feature(generic_const_items)]
#![feature(generic_const_parameter_types)]
#![feature(min_generic_const_args)]
#![feature(unsized_const_params)]
#![expect(incomplete_features)]

trait Trait {
    // NOTE: The `ConstParamTy_` bound is intentionally on the assoc const and not on the trait as
    //       doing the latter would already render the trait dyn incompatible due to it being
    //       bounded by `PartialEq<Self>` and supertrait bounds cannot mention `Self` like this.
    #[type_const]
    const K: Self where Self: std::marker::ConstParamTy_;
    //~^ NOTE it contains associated const `K` whose type references the `Self` type

    // This is not a "`Self` projection" in our sense (which would be allowed)
    // since the trait is not the principal trait or a supertrait thereof.
    #[type_const]
    const Q: <Self as SomeOtherTrait>::Output;
    //~^ NOTE it contains associated const `Q` whose type references the `Self` type
}

trait SomeOtherTrait {
    type Output: std::marker::ConstParamTy_;
}

// You could imagine this impl being more interesting and mention `T` somewhere in `Output`...
impl<T: ?Sized> SomeOtherTrait for T {
    type Output = ();
}

fn main() {
    let _: dyn Trait; //~ ERROR the trait `Trait` is not dyn compatible
}
