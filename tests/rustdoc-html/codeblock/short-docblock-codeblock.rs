#![crate_name = "foo"]

//@ count foo/index.html '//*[@class="desc docblock-short"]' 0

/// ```
/// let x = 12;
/// ```
///
/// Some text.
pub fn foo() {}
