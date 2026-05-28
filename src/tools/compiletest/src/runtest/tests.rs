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
