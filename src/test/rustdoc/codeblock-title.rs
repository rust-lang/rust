#![crate_name = "foo"]

// @has foo/fn.bar.html '//*[@class="tooltip compile_fail"]' "ⓘ"
// @has foo/fn.bar.html '//*[@class="tooltip ignore"]' "ⓘ"
// @has foo/fn.bar.html '//*[@class="tooltip should_panic"]' "ⓘ"
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
