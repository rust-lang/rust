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
/// branch regions.
///
/// FIXME(#119681): This test can be removed when we have examples of branch
/// coverage in the actual coverage test suite.
#[test]
fn anonymize_coverage_line_numbers() {
    let anon = |coverage| TestCx::anonymize_coverage_line_numbers(coverage);

    let input = r#"
    6|      3|fn print_size<T>() {
    7|      3|    if std::mem::size_of::<T>() > 4 {
  ------------------
  |  Branch (7:8): [True: 0, False: 1]
  |  Branch (7:8): [True: 0, False: 1]
  |  Branch (7:8): [True: 1, False: 0]
  ------------------
    8|      1|        println!("size > 4");
"#;

    let expected = r#"
   LL|      3|fn print_size<T>() {
   LL|      3|    if std::mem::size_of::<T>() > 4 {
  ------------------
  |  Branch (LL:8): [True: 0, False: 1]
  |  Branch (LL:8): [True: 0, False: 1]
  |  Branch (LL:8): [True: 1, False: 0]
  ------------------
   LL|      1|        println!("size > 4");
"#;

    assert_eq!(anon(input), expected);

    //////////

    let input = r#"
   12|      3|}
  ------------------
  | branch_generics::print_size::<()>:
  |    6|      1|fn print_size<T>() {
  |    7|      1|    if std::mem::size_of::<T>() > 4 {
  |  ------------------
  |  |  Branch (7:8): [True: 0, False: 1]
  |  ------------------
  |    8|      0|        println!("size > 4");
  |    9|      1|    } else {
  |   10|      1|        println!("size <= 4");
  |   11|      1|    }
  |   12|      1|}
  ------------------
"#;

    let expected = r#"
   LL|      3|}
  ------------------
  | branch_generics::print_size::<()>:
  |   LL|      1|fn print_size<T>() {
  |   LL|      1|    if std::mem::size_of::<T>() > 4 {
  |  ------------------
  |  |  Branch (LL:8): [True: 0, False: 1]
  |  ------------------
  |   LL|      0|        println!("size > 4");
  |   LL|      1|    } else {
  |   LL|      1|        println!("size <= 4");
  |   LL|      1|    }
  |   LL|      1|}
  ------------------
"#;

    assert_eq!(anon(input), expected);
}
