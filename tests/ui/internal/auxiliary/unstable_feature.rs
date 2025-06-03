#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![allow(dead_code)]
#![stable(feature = "a", since = "1.1.1" )]

#[stable(feature = "a", since = "1.1.1" )]
pub trait Foo {
    #[stable(feature = "a", since = "1.1.1" )]
    fn foo();
}
#[stable(feature = "a", since = "1.1.1" )]
pub struct Bar;

// Annotate the impl as unstable.
#[unstable_feature_bound(feat_foo)]
#[unstable(feature = "feat_foo", issue = "none" )]
impl Foo for Bar {
    fn foo() {}
}

// Use the unstable impl inside std/core.
#[unstable_feature_bound(feat_foo)]
fn bar() {
    Bar::foo();
}

fn main() {
}
