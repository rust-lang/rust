#![allow(internal_features)]
//~^ ERROR:  module has missing stability attribute
#![feature(staged_api)]
#![feature(impl_stability)]
#![allow(dead_code)]

#[stable(feature = "a", since = "1.1.1" )]
trait Moo {}
#[stable(feature = "a", since = "1.1.1" )]
trait Foo {}
#[stable(feature = "a", since = "1.1.1" )]
pub struct Bar;


// If #[unstable_feature_bound] and #[unstable] has different name,
// It should throw an error.
#[unstable(feature = "feat_moo", issue = "none" )]
#[unstable_feature_bound(feat_foo)] //~^ ERROR: an `#[unstable]` annotation here has no effect
impl Moo for Bar {}

#[unstable(feature = "feat_foo", issue = "none" )]
#[unstable_feature_bound(feat_foo)]
impl Foo for Bar {}

fn main() {}
