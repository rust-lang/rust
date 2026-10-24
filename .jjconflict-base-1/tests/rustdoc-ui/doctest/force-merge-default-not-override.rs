//@ check-pass
//@ edition: 2024
//@ compile-flags: --test --test-args=--test-threads=1 --merge-doctests=yes -Z unstable-options
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ normalize-stdout: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

// FIXME: compiletest doesn't support `// RAW` for doctests because the progress messages aren't
// emitted as JSON. Instead the .stderr file tests that this doesn't contains a
// "merged compilation took ..." message.

/// ```standalone_crate
/// let x = 12;
/// ```
///
/// These two doctests should be not be merged, even though this passes `--merge-doctests=yes`.
///
/// ```standalone_crate
/// fn main() {
///     println!("owo");
/// }
/// ```
pub struct Foo;
