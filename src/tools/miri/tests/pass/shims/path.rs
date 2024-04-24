//@compile-flags: -Zmiri-disable-isolation
use std::path::{absolute, Path};

#[track_caller]
fn test_absolute(in_: &str, out: &str) {
    assert_eq!(absolute(in_).unwrap().as_os_str(), Path::new(out).as_os_str());
}

fn main() {
    if cfg!(unix) {
        test_absolute("/a/b/c", "/a/b/c");
        test_absolute("/a/b/c", "/a/b/c");
        test_absolute("/a//b/c", "/a/b/c");
        test_absolute("//a/b/c", "//a/b/c");
        test_absolute("///a/b/c", "/a/b/c");
        test_absolute("/a/b/c/", "/a/b/c/");
        test_absolute("/a/./b/../c/.././..", "/a/b/../c/../..");
    } else if cfg!(windows) {
        // Test that all these are unchanged
        test_absolute(r"C:\path\to\file", r"C:\path\to\file");
        test_absolute(r"C:\path\to\file\", r"C:\path\to\file\");
        test_absolute(r"\\server\share\to\file", r"\\server\share\to\file");
        test_absolute(r"\\server.\share.\to\file", r"\\server.\share.\to\file");
        test_absolute(r"\\.\PIPE\name", r"\\.\PIPE\name");
        test_absolute(r"\\.\C:\path\to\COM1", r"\\.\C:\path\to\COM1");
        test_absolute(r"\\?\C:\path\to\file", r"\\?\C:\path\to\file");
        test_absolute(r"\\?\UNC\server\share\to\file", r"\\?\UNC\server\share\to\file");
        test_absolute(r"\\?\PIPE\name", r"\\?\PIPE\name");
        // Verbatim paths are always unchanged, no matter what.
        test_absolute(r"\\?\path.\to/file..", r"\\?\path.\to/file..");

        test_absolute(r"C:\path..\to.\file.", r"C:\path..\to\file");
        test_absolute(r"COM1", r"\\.\COM1");
    } else {
        panic!("unsupported OS");
    }
}
