//@ compile-flags:--test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
// Since the lint is not denied, the exit code comes from the failed doctests.
//@ failure-status: 101

/// Indented doctest which fails to compile: lint emitted.
///
///     not rust code
//~^ WARN this indented doctest failed to compile
pub struct X;

/// Fenced doctest which fails to compile: lint not emitted.
///
/// ```
/// not rust code
/// ```
pub struct Y;

/// Indented doctest which fails to compile but the lint is allowed.
///
///     not rust code
#[allow(rustdoc::unexpected_indented_doctest)]
pub struct Allowed;

/// Indented doctest which compiles: lint not emitted.
///
///     let x = 1;
pub struct Z;
