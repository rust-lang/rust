//@ compile-flags:--test
//@ failure-status: 101
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

#![doc(test(attr(deny(warnings))))]

/// ```no_run
/// let a = 3;
/// ```
pub fn foo() {}
