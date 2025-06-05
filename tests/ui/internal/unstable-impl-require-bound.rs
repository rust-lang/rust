#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![allow(dead_code)]
#![unstable(feature = "feat_foo", issue = "none" )]

// This is testing:
// 1.  Using an unstable impl requires #[unstable_feature_bound(..)]
// 2. If only feat_foo is needed to use an impl,
//    having both `feat_foo` and `feat_bar` will still make it pass.

pub trait Foo {
    fn foo();
}
pub struct Bar;

// Annotate the impl as unstable.
#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {
    fn foo() {}
}

fn bar() {
    Bar::foo();
    //~^ ERROR: cannot satisfy `unstable feature: `feat_foo``
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
