#![allow(internal_features)]
#![feature(staged_api)]
#![stable(feature = "a", since = "1.1.1" )]

#[stable(feature = "a", since = "1.1.1" )]
pub trait Foo {
    #[stable(feature = "a", since = "1.1.1" )]
    fn foo();
}
#[stable(feature = "a", since = "1.1.1" )]
pub struct Bar;
#[stable(feature = "a", since = "1.1.1" )]
pub struct Moo;

#[unstable_feature_bound(feat_bar)]
#[unstable(feature = "feat_bar", issue = "none" )]
impl Foo for Bar {
    fn foo() {}
}

#[unstable_feature_bound(feat_moo)]
#[unstable(feature = "feat_moo", issue = "none" )]
impl Foo for Moo {
    fn foo() {}
}
