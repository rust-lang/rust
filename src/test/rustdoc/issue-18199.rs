// compile-flags:--test

#![doc(test(attr(feature(staged_api))))]

/// ```
/// #![unstable(feature="test", issue="18199")]
/// fn main() {}
/// ```
pub fn foo() {}
