#![allow(internal_features)]
#![feature(staged_api)]
#![unstable(feature = "foo", issue = "none" )]

#[unstable_feature_bound(foo)]
#[unstable(feature = "foo", issue = "none" )]
pub trait Foo {
}
