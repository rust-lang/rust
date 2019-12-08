// Check that an issue value can be explicitly set to "none" instead of "0"
#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[unstable(feature = "unstable_test_feature", issue = "0")]
fn unstable_issue_0() {}

#[unstable(feature = "unstable_test_feature", issue = "none")]
fn unstable_issue_none() {}

#[unstable(feature = "unstable_test_feature", issue = "something")] //~ ERROR incorrect 'issue'
fn unstable_issue_not_allowed() {}
