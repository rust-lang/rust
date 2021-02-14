/// Doc-test test
/// ```rust
/// assert!(cargo_miri_test::make_true());
/// ```
pub fn make_true() -> bool {
    issue_1567::use_the_dependency();
    issue_1705::use_the_dependency();
    issue_1691::use_me()
}
