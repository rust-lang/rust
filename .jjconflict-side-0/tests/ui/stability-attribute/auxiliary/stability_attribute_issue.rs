#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "1.2.0")]


#[unstable(feature = "unstable_test_feature", issue = "1")]
pub fn unstable() {}

#[unstable(feature = "unstable_test_feature", reason = "message", issue = "2")]
pub fn unstable_msg() {}
