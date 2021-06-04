#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "1.0.0")]
#[deprecated] //~ ERROR `#[deprecated]` cannot be used in staged API
fn main() {}
