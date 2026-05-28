// This test checks that the test behave when `--doctest-build-arg` is passed multiple times.

//@ check-pass
//@ compile-flags: --test -Zunstable-options --doctest-build-arg=--cfg=testcase_must_be_present
//@ compile-flags: --doctest-build-arg=--cfg=another
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```
/// #[cfg(testcase_must_be_present)]
/// #[cfg(another)]
/// fn must_be_present() {}
///
/// fn main() { must_be_present() }
/// ```
pub struct Bar;
