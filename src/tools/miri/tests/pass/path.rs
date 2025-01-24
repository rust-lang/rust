//@compile-flags: -Zmiri-disable-isolation
use std::path::{Path, PathBuf, absolute};

#[path = "../utils/mod.rs"]
mod utils;

#[track_caller]
fn assert_absolute_eq(in_: &str, out: &str) {
    assert_eq!(absolute(in_).unwrap().as_os_str(), Path::new(out).as_os_str());
}

fn test_absolute() {
    if cfg!(unix) {
        assert_absolute_eq("/a/b/c", "/a/b/c");
        assert_absolute_eq("/a/b/c", "/a/b/c");
        assert_absolute_eq("/a//b/c", "/a/b/c");
        assert_absolute_eq("//a/b/c", "//a/b/c");
        assert_absolute_eq("///a/b/c", "/a/b/c");
        assert_absolute_eq("/a/b/c/", "/a/b/c/");
        assert_absolute_eq("/a/./b/../c/.././..", "/a/b/../c/../..");
    } else if cfg!(windows) {
        // Test that all these are unchanged
        assert_absolute_eq(r"C:\path\to\file", r"C:\path\to\file");
        assert_absolute_eq(r"C:\path\to\file\", r"C:\path\to\file\");
        assert_absolute_eq(r"\\server\share\to\file", r"\\server\share\to\file");
        assert_absolute_eq(r"\\server.\share.\to\file", r"\\server.\share.\to\file");
        assert_absolute_eq(r"\\.\PIPE\name", r"\\.\PIPE\name");
        assert_absolute_eq(r"\\.\C:\path\to\COM1", r"\\.\C:\path\to\COM1");
        assert_absolute_eq(r"\\?\C:\path\to\file", r"\\?\C:\path\to\file");
        assert_absolute_eq(r"\\?\UNC\server\share\to\file", r"\\?\UNC\server\share\to\file");
        assert_absolute_eq(r"\\?\PIPE\name", r"\\?\PIPE\name");
        // Verbatim paths are always unchanged, no matter what.
        assert_absolute_eq(r"\\?\path.\to/file..", r"\\?\path.\to/file..");

        assert_absolute_eq(r"C:\path..\to.\file.", r"C:\path..\to\file");
        assert_absolute_eq(r"COM1", r"\\.\COM1");
    } else {
        panic!("unsupported OS");
    }
}

fn buf_smoke(mut p: PathBuf) {
    for _c in p.components() {}

    p.push("hello");
    for _c in p.components() {}

    if cfg!(windows) {
        p.push(r"C:\mydir");
    } else {
        p.push(r"/mydir");
    }
    for _c in p.components() {}
}

fn main() {
    buf_smoke(PathBuf::new());
    buf_smoke(utils::tmp());
    test_absolute();
}
