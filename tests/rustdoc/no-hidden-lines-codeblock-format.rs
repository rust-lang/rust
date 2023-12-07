// Test that the `no_hidden_lines` codeblock attribute prevents lines starting with `#` to
// be stripped.

#![crate_name = "foo"]

// @has 'foo/index.html'
// @matches - '//*[@class="rust rust-example-rendered"]/code' '\(\s+# a,\s+##b,\s+###c\s+\);'
// @snapshot 'codeblock' - '//*[@class="rust rust-example-rendered"]/code'

//! ```rust,no_hidden_lines
//! macro_rules! test {
//!     (# a,##b,###c) => {}
//! }
//!
//! test!(
//!     # a,
//!     ##b,
//!     ###c
//! );
//! ```

// @has 'foo/fn.foo.html'
// @matches - '//*[@class="rust rust-example-rendered"]/code' '\(\s+#b,\s+##c\s+\);'
// @snapshot 'codeblock-non-raw' - '//*[@class="rust rust-example-rendered"]/code'

/// ```
/// macro_rules! test {
///     (# a,##b,###c) => {}
/// }
///
/// test!(
///     # a,
///     ##b,
///     ###c
/// );
/// ```
pub fn foo() {}

// Testing that `raw` means that it is a rust code block by default.
// @has 'foo/fn.bar.html'
// @matches - '//*[@class="rust rust-example-rendered"]/code' '\(\s+#a,\s+##b,\s+###c\s+\);'

/// ```no_hidden_lines
/// macro_rules! test {
///     (#a,##b,###c) => {}
/// }
///
/// test!(
///     #a,
///     ##b,
///     ###c
/// );
/// ```
pub fn bar() {}
