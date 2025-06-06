#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![allow(dead_code)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// In staged-api crate, impl that is marked with `feat_moo`
/// should not be accessible if only `feat_foo` is enabled.

/// FIXME: add one more check pass test with revision?

pub trait Foo {
    fn foo();
}

pub trait Moo {
    fn moo();
}

pub struct Bar;

#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {
    fn foo() {}
}

#[unstable_feature_bound(feat_moo)]
impl Moo for Bar {
    fn moo() {}
}

#[unstable_feature_bound(feat_foo)]
fn bar() {
    Bar::foo();
    Bar::moo();
    //~^ ERROR cannot satisfy `unstable feature: `feat_moo``
}

fn main() {}
