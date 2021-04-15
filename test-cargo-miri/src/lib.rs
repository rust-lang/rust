extern crate exported_symbol;

/// Doc-test test
/// ```rust
/// assert!(cargo_miri_test::make_true());
/// // Repeat calls to make sure the `Instance` cache is not broken.
/// for _ in 0..3 {
///     extern "Rust" {
///         fn exported_symbol() -> i32;
///         fn make_true() -> bool;
///     }
///     assert_eq!(unsafe { exported_symbol() }, 123456);
///     assert!(unsafe { make_true() });
/// }
/// ```
/// ```compile_fail
/// // Make sure `exported_symbol_dep` is not a direct dependency for doctests.
/// use exported_symbol_dep;
/// ```
/// ```rust,no_run
/// assert!(!cargo_miri_test::make_true());
/// ```
/// ```rust,compile_fail
/// assert!(cargo_miri_test::make_true() == 5);
/// ```
#[no_mangle]
pub fn make_true() -> bool {
    issue_1567::use_the_dependency();
    issue_1705::use_the_dependency();
    issue_1760::use_the_dependency!();
    issue_1691::use_me()
}

pub fn main() {
    println!("imported main");
}
