#![allow(internal_features)]
#![feature(staged_api)]
#![allow(dead_code)]
#![stable(feature = "a", since = "1.1.1" )]

// Lint against the usage of both #[unstable_feature_bound] and #[stable] on the
// same item.

#[stable(feature = "a", since = "1.1.1")]
#[unstable_feature_bound(feat_bar)]
fn bar() {}
//~^ ERROR item annotated with `#[unstable_feature_bound]` should not be stable

fn main() {}
