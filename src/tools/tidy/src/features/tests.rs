use super::*;

#[test]
fn test_find_attr_val() {
    let s = r#"#[unstable(feature = "tidy_test_never_used_anywhere_else", issue = "58402")]"#;
    assert_eq!(find_attr_val(s, "feature"), Some("tidy_test_never_used_anywhere_else"));
    assert_eq!(find_attr_val(s, "issue"), Some("58402"));
    assert_eq!(find_attr_val(s, "since"), None);
}
