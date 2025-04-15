//@ check-pass
//@ compile-flags: --test -Zunstable-options --doctest-build-arg=--cfg=testcase_must_be_present
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```
/// #[cfg(testcase_must_be_present)]
/// fn must_be_present() {}
///
/// fn main() { must_be_present() }
/// ```
pub struct Bar;
