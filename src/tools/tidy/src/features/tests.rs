use super::*;

#[test]
fn test_find_attr_val() {
    let s = r#"#[unstable(feature = "checked_duration_since", issue = "58402")]"#;
    assert_eq!(find_attr_val(s, "feature"), Some("checked_duration_since"));
    assert_eq!(find_attr_val(s, "issue"), Some("58402"));
    assert_eq!(find_attr_val(s, "since"), None);
}
