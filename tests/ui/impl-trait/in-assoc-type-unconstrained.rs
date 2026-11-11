#![feature(impl_trait_in_assoc_type)]

mod compare_ty {
    trait Trait {
        type Ty: IntoIterator<Item = ()>;
    }
    impl Trait for () {
        type Ty = Option<impl Sized>;
        //~^ ERROR: unconstrained opaque type
        //~| ERROR: type mismatch resolving `<Option<<() as Trait>::Ty::{opaque#0}> as IntoIterator>::Item == ()`
    }
}

mod compare_method {
    trait Trait {
        type Ty;
        fn method() -> Self::Ty;
    }
    impl Trait for () {
        type Ty = impl Sized;
        //~^ ERROR: unconstrained opaque type
        fn method() -> () {}
        //~^ ERROR: method `method` has an incompatible type for trait
    }
}

fn main() {}
