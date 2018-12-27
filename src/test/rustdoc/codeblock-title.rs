#![crate_name = "foo"]

// ignore-tidy-linelength

// @has foo/fn.bar.html '//*[@class="tooltip compile_fail"]/span' "This example deliberately fails to compile"
// @has foo/fn.bar.html '//*[@class="tooltip ignore"]/span' "This example is not tested"

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
/// ```
/// let x = 0;
/// ```
pub fn bar() -> usize { 2 }
