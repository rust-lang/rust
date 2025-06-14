//@ needs-target-std
use run_make_support::{rfs, rustc};

fn main() {
    #[cfg(unix)]
    let non_unicode: &std::ffi::OsStr = std::os::unix::ffi::OsStrExt::from_bytes(&[0xFF]);
    #[cfg(windows)]
    let non_unicode: std::ffi::OsString = std::os::windows::ffi::OsStringExt::from_wide(&[0xD800]);
    let output = rustc().input("non_unicode_env.rs").env("NON_UNICODE_VAR", non_unicode).run_fail();
    let expected = rfs::read_to_string("non_unicode_env.stderr");
    output.assert_stderr_equals(expected);
}
