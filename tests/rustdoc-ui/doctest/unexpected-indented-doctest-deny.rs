//@ compile-flags:--test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
// If the lint is denied, the exit code is not 101 but 1.
//@ failure-status: 1

#![deny(rustdoc::unexpected_indented_doctest)]

/// text
///
///     not rust code
//~^ ERROR this indented doctest failed to compile
pub struct X;
