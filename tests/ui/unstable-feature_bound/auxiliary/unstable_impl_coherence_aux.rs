#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![stable(feature = "a", since = "1.1.1" )]

#[stable(feature = "a", since = "1.1.1" )]
pub trait Trait {}

#[unstable_feature_bound(foo)]
#[unstable(feature = "foo", issue = "none" )]
impl <T> Trait for T {}
