//@ compile-flags:--test
//@ should-fail

#![doc(test(attr(deny(warnings))))]

/// ```no_run
/// let a = 3;
/// ```
pub fn foo() {}
