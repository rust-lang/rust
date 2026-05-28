//@ compile-flags:--test
// https://github.com/rust-lang/rust/issues/18199

#![doc(test(attr(feature(staged_api))))]

/// ```
/// #![allow(internal_features)]
/// #![unstable(feature="test", issue="18199")]
/// fn main() {}
/// ```
pub fn foo() {}
