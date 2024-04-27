use super::*;

#[test]
fn normalize_platform_differences() {
    assert_eq!(TestCx::normalize_platform_differences(r"$DIR\foo.rs"), "$DIR/foo.rs");
    assert_eq!(
        TestCx::normalize_platform_differences(r"$BUILD_DIR\..\parser.rs"),
        "$BUILD_DIR/../parser.rs"
    );
    assert_eq!(
        TestCx::normalize_platform_differences(r"$DIR\bar.rs: hello\nworld"),
        r"$DIR/bar.rs: hello\nworld"
    );
    assert_eq!(
        TestCx::normalize_platform_differences(r"either bar\baz.rs or bar\baz\mod.rs"),
        r"either bar/baz.rs or bar/baz/mod.rs",
    );
    assert_eq!(TestCx::normalize_platform_differences(r"`.\some\path.rs`"), r"`./some/path.rs`",);
    assert_eq!(TestCx::normalize_platform_differences(r"`some\path.rs`"), r"`some/path.rs`",);
    assert_eq!(
        TestCx::normalize_platform_differences(r"$DIR\path-with-dashes.rs"),
        r"$DIR/path-with-dashes.rs"
    );
    assert_eq!(
        TestCx::normalize_platform_differences(r"$DIR\path_with_underscores.rs"),
        r"$DIR/path_with_underscores.rs",
    );
    assert_eq!(TestCx::normalize_platform_differences(r"$DIR\foo.rs:12:11"), "$DIR/foo.rs:12:11",);
    assert_eq!(
        TestCx::normalize_platform_differences(r"$DIR\path with\spaces 'n' quotes"),
        "$DIR/path with/spaces 'n' quotes",
    );
    assert_eq!(
        TestCx::normalize_platform_differences(r"$DIR\file_with\no_extension"),
        "$DIR/file_with/no_extension",
    );

    assert_eq!(TestCx::normalize_platform_differences(r"\n"), r"\n");
    assert_eq!(TestCx::normalize_platform_differences(r"{ \n"), r"{ \n");
    assert_eq!(TestCx::normalize_platform_differences(r"`\]`"), r"`\]`");
    assert_eq!(TestCx::normalize_platform_differences(r#""\{""#), r#""\{""#);
    assert_eq!(
        TestCx::normalize_platform_differences(r#"write!(&mut v, "Hello\n")"#),
        r#"write!(&mut v, "Hello\n")"#
    );
    assert_eq!(
        TestCx::normalize_platform_differences(r#"println!("test\ntest")"#),
        r#"println!("test\ntest")"#,
    );
}

/// Test for anonymizing line numbers in coverage reports, especially for
/// MC/DC regions.
///
/// FIXME(#123409): This test can be removed when we have examples of MC/DC
/// coverage in the actual coverage test suite.
#[test]
fn anonymize_coverage_line_numbers() {
    let anon = |coverage| TestCx::anonymize_coverage_line_numbers(coverage);

    let input = r#"
    7|      2|fn mcdc_check_neither(a: bool, b: bool) {
    8|      2|    if a && b {
                          ^0
  ------------------
  |---> MC/DC Decision Region (8:8) to (8:14)
  |
  |  Number of Conditions: 2
  |     Condition C1 --> (8:8)
  |     Condition C2 --> (8:13)
  |
  |  Executed MC/DC Test Vectors:
  |
  |     C1, C2    Result
  |  1 { F,  -  = F      }
  |
  |  C1-Pair: not covered
  |  C2-Pair: not covered
  |  MC/DC Coverage for Decision: 0.00%
  |
  ------------------
    9|      0|        say("a and b");
   10|      2|    } else {
   11|      2|        say("not both");
   12|      2|    }
   13|      2|}
"#;

    let expected = r#"
   LL|      2|fn mcdc_check_neither(a: bool, b: bool) {
   LL|      2|    if a && b {
                          ^0
  ------------------
  |---> MC/DC Decision Region (LL:8) to (LL:14)
  |
  |  Number of Conditions: 2
  |     Condition C1 --> (LL:8)
  |     Condition C2 --> (LL:13)
  |
  |  Executed MC/DC Test Vectors:
  |
  |     C1, C2    Result
  |  1 { F,  -  = F      }
  |
  |  C1-Pair: not covered
  |  C2-Pair: not covered
  |  MC/DC Coverage for Decision: 0.00%
  |
  ------------------
   LL|      0|        say("a and b");
   LL|      2|    } else {
   LL|      2|        say("not both");
   LL|      2|    }
   LL|      2|}
"#;

    assert_eq!(anon(input), expected);
}
