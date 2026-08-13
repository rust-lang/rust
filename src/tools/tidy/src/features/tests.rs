use super::*;

#[test]
fn test_find_attr_val() {
    let s = r#"#[unstable(feature = "tidy_test_never_used_anywhere_else", issue = "58402")]"#;
    assert_eq!(find_attr_val(s, "feature"), Some("tidy_test_never_used_anywhere_else"));
    assert_eq!(find_attr_val(s, "issue"), Some("58402"));
    assert_eq!(find_attr_val(s, "since"), None);
}

#[track_caller]
fn check_extract_lib_features(contents: &str, expected: &[(Result<(), &str>, usize)]) {
    let mut expected = expected.iter().cloned().collect::<Vec<_>>();
    expected.sort_unstable_by_key(|(_, line)| *line);

    let mut found = Vec::with_capacity(expected.len());
    extract_lib_features(contents, Path::new(""), &mut |result, _, line| {
        found.push((result.map(drop), line))
    });
    expected.sort_unstable_by_key(|(_, line)| *line);

    assert_eq!(expected, found);
}

#[test]
fn extract_lib_features_invalid() {
    check_extract_lib_features(
        r#"
#[stable]
#![stable]
    // #[stable]
#[stable(feature = "foo")]
#[stable(since = "1.97.0")]
#[stable(feature = "foo", since = "something")]
#[unstable(issue = "foo")]
#[rustc_const_unstable(issue = "foo")]
// #[unstable(feature = "foo")] // FIXME: Is not putting `issue` really fine?
    "#,
        &[
            (Err("malformed stability attribute: missing `feature` key"), 2),
            (Err("malformed stability attribute: missing `feature` key"), 3),
            (Err("malformed stability attribute: missing the `since` key"), 5),
            (Err("malformed stability attribute: missing `feature` key"), 6),
            (Err("malformed stability attribute: can't parse `since` key"), 7),
            (Err("malformed stability attribute: missing `feature` key"), 8),
            (Err("malformed stability attribute: missing `feature` key"), 9),
        ],
    );
}

#[test]
fn extract_lib_features_invalid_multiline() {
    check_extract_lib_features(
        r#"
// #[stable(
// )]
    #[stable(
        feature = "windows_process_extensions_main_thread_handle",
        since = "CURRENT_RUSTC_VERSION"
    )]
    "#,
        &[(Ok(()), 4)],
    );
}
