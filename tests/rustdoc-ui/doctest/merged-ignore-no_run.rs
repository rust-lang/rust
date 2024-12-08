//@ edition: 2024
//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// ```ignore (test)
/// let x = 12;
/// ```
pub fn ignored() {}

/// ```no_run
/// panic!("blob");
/// ```
pub fn no_run() {}
