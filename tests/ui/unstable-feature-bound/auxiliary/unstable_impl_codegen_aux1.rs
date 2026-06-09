#![allow(internal_features)]
#![feature(staged_api)]
#![stable(feature = "a", since = "1.1.1" )]

/// Aux crate for unstable impl codegen test.

#[stable(feature = "a", since = "1.1.1" )]
pub trait Trait {
    #[stable(feature = "a", since = "1.1.1" )]
    fn method(&self);
}

#[unstable_feature_bound(foo)]
#[unstable(feature = "foo", issue = "none" )]
impl<T> Trait for T {
    fn method(&self) {
        println!("hi");
    }
}
