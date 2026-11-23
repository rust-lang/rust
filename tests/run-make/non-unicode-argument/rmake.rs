// Regression test for https://github.com/rust-lang/rust/issues/164585

use run_make_support::rustc;

fn main() {
    #[cfg(unix)]
    let non_unicode: &std::ffi::OsStr = std::os::unix::ffi::OsStrExt::from_bytes(&[0xFF]);
    #[cfg(windows)]
    let non_unicode: std::ffi::OsString = std::os::windows::ffi::OsStringExt::from_wide(&[0xD800]);

    rustc()
        .env_remove("RUST_BACKTRACE")
        .arg(&non_unicode)
        .run_fail()
        .assert_exit_code(1)
        .assert_stderr_contains("is not valid Unicode")
        .assert_stderr_not_contains("panicked at");

    rustc()
        .env("RUST_BACKTRACE", "0")
        .arg(&non_unicode)
        .run_fail()
        .assert_exit_code(1)
        .assert_stderr_contains("is not valid Unicode");
}
