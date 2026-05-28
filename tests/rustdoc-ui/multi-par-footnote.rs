//@ check-pass
//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
// Regression test for #139064.

/// Example
///
/// Footnote with multiple paragraphs[^multiple]
///
/// [^multiple]:
///     One
///
///     Two
///
///     Three
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
