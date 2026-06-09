//@ revisions: pass fail
//@[pass] check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![unstable(feature = "feat_foo", issue = "none" )]

#![cfg_attr(fail, feature(feat_foo))]

/// In staged-api crate, using an unstable impl requires
/// #[unstable_feature_bound(..)], not  #[feature(..)].

pub trait Foo {
    fn foo();
}
pub struct Bar;

#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {
    fn foo() {}
}

#[cfg_attr(pass, unstable_feature_bound(feat_foo))]
fn bar() {
    Bar::foo();
    //[fail]~^ ERROR: unstable feature `feat_foo` is used without being enabled.
}

fn main() {}
