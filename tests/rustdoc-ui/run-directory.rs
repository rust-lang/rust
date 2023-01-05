// this test asserts that the cwd of doctest invocations is set correctly.

// revisions: correct incorrect
// check-pass
// [correct]compile-flags:--test --test-run-directory={{src-base}} -Zunstable-options
// [incorrect]compile-flags:--test --test-run-directory={{src-base}}/coverage -Zunstable-options
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```
/// assert_eq!(
///     std::fs::read_to_string("run-directory.rs").unwrap(),
///     include_str!("run-directory.rs"),
/// );
/// ```
#[cfg(correct)]
pub fn foo() {}

/// ```
/// assert!(std::fs::read_to_string("run-directory.rs").is_err());
/// ```
#[cfg(incorrect)]
pub fn foo() {}
