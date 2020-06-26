#![crate_name = "foo"]

// ignore-tidy-linelength

// @has foo/fn.bar.html '//*[@class="tooltip compile_fail"]/span' "This example deliberately fails to compile"
// @has foo/fn.bar.html '//*[@class="tooltip ignore"]/span' "This example is not tested"
// @has foo/fn.bar.html '//*[@class="tooltip should_panic"]/span' "This example panics"

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
/// ```
/// let x = 0;
/// ```
pub fn bar() -> usize { 2 }
