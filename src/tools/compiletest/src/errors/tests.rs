use super::*;

#[test]
fn test_parse_expected_matching() {
    // Ensure that we correctly extract expected revisions
    let d1 = "//[rev1,rev2]~^ ERROR foo";
    let d2 = "//[rev1,rev2-foo]~^ ERROR foo";
    assert!(parse_expected(None, 1, d1, Some("rev1")).is_some());
    assert!(parse_expected(None, 1, d1, Some("rev2")).is_some());
    assert!(parse_expected(None, 1, d2, Some("rev1")).is_some());
    assert!(parse_expected(None, 1, d2, Some("rev2-foo")).is_some());
}
