//@ revisions: pass fail 
//@[pass] check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![allow(dead_code)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// In staged-api crate, using an unstable impl 
/// requires #[unstable_feature_bound(..)].

pub trait Foo {
    fn foo();
}
pub struct Bar;

// Annotate the impl as unstable.
#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {
    fn foo() {}
}

#[cfg_attr(pass, unstable_feature_bound(feat_foo))]
fn bar() {
    Bar::foo();
    //[fail]~^ ERROR: cannot satisfy `unstable feature: `feat_foo``
}

// With #[unstable_feature_bound(..)], this should pass.
#[unstable_feature_bound(feat_foo)]
fn bar2() {
    Bar::foo();
}

// FIXME: maybe we should warn that feat_bar does not exist?
#[unstable_feature_bound(feat_foo, feat_bar)]
fn bar3() {
    Bar::foo();
}

fn main() {}
