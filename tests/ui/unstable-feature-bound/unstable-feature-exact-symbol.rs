//@ revisions: pass fail
//@[pass] check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// In staged-api crate, impl that is marked as unstable with
/// feature name `feat_moo` should not be accessible
/// if only `feat_foo` is enabled.

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

#[cfg_attr(fail, unstable_feature_bound(feat_foo))]
#[cfg_attr(pass, unstable_feature_bound(feat_foo, feat_moo))]
fn bar() {
    Bar::foo();
    Bar::moo();
    //[fail]~^ ERROR  unstable feature `feat_moo` is used without being enabled.

}

fn main() {}
