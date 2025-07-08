#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![stable(feature = "a", since = "1.1.1" )]

// Test the behaviour of putting an #[unstable_feature_bound] on a stable function.

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


#[stable(feature = "a", since = "1.1.1")]
#[unstable_feature_bound(feat_bar)]
fn bar() {
    Bar::foo();
}

fn main() {}
