//@ edition: 2024
//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// ```ignore (test)
/// let x = 12;
/// ```
pub fn ignored() {}

/// ```no_run
/// panic!("blob");
/// ```
pub fn no_run() {}
