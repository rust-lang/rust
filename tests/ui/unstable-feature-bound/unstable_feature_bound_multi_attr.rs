#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![unstable(feature = "feat_bar", issue = "none" )]

/// Test the behaviour of multiple unstable_feature_bound attribute.

trait Foo {
    fn foo();
}
struct Bar;

#[unstable_feature_bound(feat_bar, feat_koo)]
#[unstable_feature_bound(feat_foo, feat_moo)]
impl Foo for Bar {
    fn foo(){}
}

#[unstable_feature_bound(feat_bar, feat_koo)]
#[unstable_feature_bound(feat_foo, feat_moo)]
fn moo() {
    Bar::foo();
}

#[unstable_feature_bound(feat_bar, feat_koo, feat_foo, feat_moo)]
fn koo() {
    Bar::foo();
}

#[unstable_feature_bound(feat_koo, feat_foo, feat_moo)]
fn boo() {
    Bar::foo();
    //~^ ERROR: unstable feature `feat_bar` is used without being enabled.
}

fn main() {}
