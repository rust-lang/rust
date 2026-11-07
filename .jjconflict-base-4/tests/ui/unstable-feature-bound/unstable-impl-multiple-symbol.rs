//@ check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// In staged-api crate, if feat_foo is only needed to use an impl,
/// having both `feat_foo` and `feat_bar` will still make it pass.

pub trait Foo {
    fn foo();
}
pub struct Bar;

// Annotate the impl as unstable.
#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {
    fn foo() {}
}

#[unstable_feature_bound(feat_foo, feat_bar)]
fn bar() {
    Bar::foo();
}

fn main() {}
