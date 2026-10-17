#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![stable(feature = "a", since = "1.1.1" )]

/// If #[unstable(..)] and #[unstable_feature_name(..)] have the same feature name,
/// the error should not be thrown as it can effectively mark an impl as unstable.
///
/// If the feature name in #[feature] does not exist in #[unstable_feature_bound(..)]
/// an error should still be thrown because that feature will not be unstable.

#[stable(feature = "a", since = "1.1.1")]
trait Moo {}
#[stable(feature = "a", since = "1.1.1")]
trait Foo {}
#[stable(feature = "a", since = "1.1.1")]
trait Boo {}
#[stable(feature = "a", since = "1.1.1")]
pub struct Bar;


#[unstable(feature = "feat_moo", issue = "none")]
#[unstable_feature_bound(feat_foo)] //~^ ERROR: an `#[unstable]` annotation here has no effect
impl Moo for Bar {}

#[unstable(feature = "feat_foo", issue = "none")]
#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {}


#[unstable(feature = "feat_foo", issue = "none")]
#[unstable_feature_bound(feat_foo, feat_bar)]
impl Boo for Bar {}

fn main() {}
