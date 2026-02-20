use super::*;

#[test]
fn path_buf_with_extra_extension_test() {
    assert_eq!(
        Utf8PathBuf::from("foo.rs.stderr"),
        Utf8PathBuf::from("foo.rs").with_extra_extension("stderr")
    );
    assert_eq!(
        Utf8PathBuf::from("foo.rs.stderr"),
        Utf8PathBuf::from("foo.rs").with_extra_extension(".stderr")
    );
    assert_eq!(Utf8PathBuf::from("foo.rs"), Utf8PathBuf::from("foo.rs").with_extra_extension(""));
}

#[test]
fn env_var_is_set_returns_true_for_non_empty_value() {
    let name = "COMPILETEST_TEST_ENV_VAR_IS_SET_TRUE";
    // SAFETY: test-only, not running concurrent tests that depend on this var.
    unsafe { std::env::set_var(name, "1") };
    assert!(env_var_is_set(name));
    unsafe { std::env::remove_var(name) };
}

#[test]
fn env_var_is_set_returns_false_when_unset() {
    let name = "COMPILETEST_TEST_ENV_VAR_IS_SET_UNSET";
    // SAFETY: test-only, not running concurrent tests that depend on this var.
    unsafe { std::env::remove_var(name) };
    assert!(!env_var_is_set(name));
}

#[test]
fn env_var_is_set_returns_false_for_empty_value() {
    let name = "COMPILETEST_TEST_ENV_VAR_IS_SET_EMPTY";
    // SAFETY: test-only, not running concurrent tests that depend on this var.
    unsafe { std::env::set_var(name, "") };
    assert!(!env_var_is_set(name));
    unsafe { std::env::remove_var(name) };
}
