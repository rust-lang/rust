//@ revisions: pass fail
//@[pass] check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![stable(feature = "a", since = "1.1.1" )]

/// When a free function with #[unstable_feature_bound(feat_bar)] is called by another
/// free function, that function should be annotated with
/// #[unstable_feature_bound(feat_bar)] too.

#[stable(feature = "a", since = "1.1.1")]
trait Foo {
    #[stable(feature = "a", since = "1.1.1")]
    fn foo() {
    }
}
#[stable(feature = "a", since = "1.1.1")]
pub struct Bar;

#[unstable_feature_bound(feat_bar)]
#[unstable(feature = "feat_bar", issue = "none" )]
impl Foo for Bar {
    fn foo() {}
}


#[unstable_feature_bound(feat_bar)]
fn bar() {
    Bar::foo();
}

#[cfg_attr(pass, unstable_feature_bound(feat_bar))]
fn bar2() {
    bar();
    //[fail]~^ERROR unstable feature `feat_bar` is used without being enabled.
}

fn main() {}
