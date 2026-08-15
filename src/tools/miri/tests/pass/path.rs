//@compile-flags: -Zmiri-disable-isolation
use std::path::{Path, PathBuf, absolute};

#[path = "../utils/mod.rs"]
mod utils;

#[track_caller]
fn assert_absolute_eq(in_: &str, out: &str) {
    assert_eq!(
        absolute(in_).unwrap().as_os_str(),
        Path::new(out).as_os_str(),
        "incorrect absolute path for {in_:?}"
    );
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
        assert_absolute_eq(r"\\server\share\NUL", r"\\server\share\NUL");
        // This fails on Windows 10 hosts. FIXME: enable this once GHA runners are on Windows 11.
        //assert_absolute_eq(r"C:\path\to\COM1", r"C:\path\to\COM1");
        // Verbatim paths are always unchanged, no matter what.
        assert_absolute_eq(r"\\?\path.\to/file..", r"\\?\path.\to/file..");
        // Trailing dot is removed here.
        assert_absolute_eq(r"C:\path..\to.\file.", r"C:\path..\to\file");
        // `..` is resolved here.
        assert_absolute_eq(r"C:\path\to\..\file", r"C:\path\file");
        assert_absolute_eq(r"C:\path\to\..\..\file", r"C:\file");
        assert_absolute_eq(r"C:\path\to\..\..\..\..\..\..\file", r"C:\file");
        assert_absolute_eq(r"C:\..", r"C:\");
        assert_absolute_eq(r"\\server\share\to\path\with\..\file", r"\\server\share\to\path\file");
        assert_absolute_eq(r"\\server\share\to\..\..\..\..\file", r"\\server\share\file");
        assert_absolute_eq(r"\\server\share\..", r"\\server\share");
        // Magic filenames.
        assert_absolute_eq(r"NUL", r"\\.\NUL");
        assert_absolute_eq(r"nul", r"\\.\nul");
        assert_absolute_eq(r"COM1", r"\\.\COM1");
        assert_absolute_eq(r"com1", r"\\.\com1");
        assert_absolute_eq(r"C:\path\to\NUL", r"\\.\NUL");
        assert_absolute_eq(r"C:\path\to\nul", r"\\.\nul");
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
