// https://github.com/rust-lang/rust/issues/18199

//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

#![doc(test(attr(feature(staged_api))))]

/// ```
/// #![allow(internal_features)]
/// #![unstable(feature="test", issue="18199")]
/// fn main() {}
/// ```
pub fn foo() {}
