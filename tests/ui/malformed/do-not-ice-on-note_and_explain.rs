struct A<B>(B);
impl<B>A<B>{fn d(){fn d(){Self(1)}}}
//~^ ERROR the size for values of type `B` cannot be known at compilation time
//~| ERROR the size for values of type `B` cannot be known at compilation time
//~| ERROR mismatched types
//~| ERROR mismatched types
//~| ERROR `main` function not found in crate
