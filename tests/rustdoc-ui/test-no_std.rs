// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
// check-pass

#![no_std]

extern crate alloc;

/// ```
/// assert!(true)
/// ```
pub fn f() {}
