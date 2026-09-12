#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "unstable_glob_source_crate", since = "1.0.0")]

#[unstable(feature = "unstable_glob_source", issue = "none")]
pub fn unstable_a() {}

#[unstable(
    feature = "unstable_glob_source",
    reason = "different reason",
    issue = "none"
)]
pub fn unstable_b() {}
