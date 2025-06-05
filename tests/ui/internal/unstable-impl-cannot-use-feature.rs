#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![allow(dead_code)]
#![feature(feat_foo)]
#![unstable(feature = "feat_foo", issue = "none" )]

/// In staged-api crate, impl that is marked with #[unstable_feature_bound(..)]
/// cannot be enabled with #[feature(..)]

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

fn main() {}
