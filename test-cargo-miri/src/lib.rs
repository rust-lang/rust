/// Doc-test test
/// ```rust
/// assert!(cargo_miri_test::make_true());
/// ```
pub fn make_true() -> bool {
    rlib_dep::use_me()
}
