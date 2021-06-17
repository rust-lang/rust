#![crate_name = "foo"]

// @has foo/index.html '//*[@class="item-right docblock-short"]' ""
// @!has foo/index.html '//*[@class="item-right docblock-short"]' "Some text."
// @!has foo/index.html '//*[@class="item-right docblock-short"]' "let x = 12;"

/// ```
/// let x = 12;
/// ```
///
/// Some text.
pub fn foo() {}
