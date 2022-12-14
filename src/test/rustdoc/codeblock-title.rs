#![crate_name = "foo"]

// @has foo/fn.bar.html '//*[@class="example-wrap compile_fail"]/*[@class="tooltip"]' "ⓘ"
// @has foo/fn.bar.html '//*[@class="example-wrap ignore"]/*[@class="tooltip"]' "ⓘ"
// @has foo/fn.bar.html '//*[@class="example-wrap should_panic"]/*[@class="tooltip"]' "ⓘ"
// @has foo/fn.bar.html '//*[@data-edition="2018"]' "ⓘ"

/// foo
///
/// ```compile_fail
/// foo();
/// ```
///
/// ```ignore (tidy)
/// goo();
/// ```
///
/// ```should_panic
/// hoo();
/// ```
///
/// ```edition2018
/// let x = 0;
/// ```
pub fn bar() -> usize { 2 }
