// rustfmt-wrap_comments: true

/// Foo
///
/// # Example
/// ```
/// # #![cfg_attr(not(dox), feature(cfg_target_feature, target_feature, stdsimd))]
/// # #![cfg_attr(not(dox), no_std)]
/// fn foo() {  }
/// ```
fn foo() {}

/// A long comment for wrapping
/// This is a long long long long long long long long long long long long long
/// long long long long long long long sentence.
fn bar() {}
