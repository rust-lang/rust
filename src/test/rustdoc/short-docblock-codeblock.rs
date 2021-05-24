#![crate_name = "foo"]

// @has foo/index.html '//*[@class="module-item"]//following-sibling::item-right[@class="docblock-short"]' ""
// @!has foo/index.html '//*[@id="module-item"]//following-sibling::item-right[@class="docblock-short"]' "Some text."
// @!has foo/index.html '//*[@id="module-item"]//following-sibling::item-right[@class="docblock-short"]' "let x = 12;"

/// ```
/// let x = 12;
/// ```
///
/// Some text.
pub fn foo() {}
