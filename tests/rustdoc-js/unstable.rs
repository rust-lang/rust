#![feature(staged_api)]
#![stable(feature = "another", since = "1.0.0")]

#[unstable(feature = "tadam", issue = "none")]
pub fn bar1() {}

#[stable(feature = "another", since = "1.0.0")]
pub fn bar2() {}
