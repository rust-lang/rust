#![allow(internal_features)] // Enabled to use #![feature(staged_api)] and #![feature(impl_stability)]
#![feature(staged_api)] // Enabled to use  #![unstable(feature = "feat_foo", issue = "none")]
#![feature(impl_stability)] // Enabled to use #[unstable_feature_bound(feat_foo)] 
#![unstable(feature = "feat_foo", issue = "none")]

pub trait Foo {
    fn foo();
}
pub struct Bar;

// Annotate the impl as unstable.
#[unstable_feature_bound(feat_foo)] 
impl Foo for Bar {
    fn foo() {}
}

// Use the unstable impl inside std/core.
#[unstable_feature_bound(feat_foo)] 
#[allow(dead_code)]
fn bar() {
    Bar::foo();
}

fn main() {
}
