#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "lint_stability", since = "1.0.0")]

extern crate lint_stability;

// Re-exporting without enabling the feature "unstable_test_feature" in this crate
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub use lint_stability::unstable_text;
