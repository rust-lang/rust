import core::*;

use std;
import std::fs;
import vec;

#[test]
fn test_connect() {
    let slash = fs::path_sep();
    log(error, fs::connect("a", "b"));
    assert (fs::connect("a", "b") == "a" + slash + "b");
    assert (fs::connect("a" + slash, "b") == "a" + slash + "b");
}

// Issue #712
#[test]
fn test_list_dir_no_invalid_memory_access() { fs::list_dir("."); }

#[test]
fn list_dir() {
    let dirs = fs::list_dir(".");
    // Just assuming that we've got some contents in the current directory
    assert (vec::len(dirs) > 0u);

    for dir in dirs { log(debug, dir); }
}

#[test]
fn path_is_dir() {
    assert (fs::path_is_dir("."));
    assert (!fs::path_is_dir("test/stdtest/fs.rs"));
}

#[test]
fn path_exists() {
    assert (fs::path_exists("."));
    assert (!fs::path_exists("test/nonexistent-bogus-path"));
}

fn ps() -> str {
    fs::path_sep()
}

fn aps() -> str {
    "/"
}

#[test]
fn split1() {
    let actual = fs::split("a" + ps() + "b");
    let expected = ["a", "b"];
    assert actual == expected;
}

#[test]
fn split2() {
    let actual = fs::split("a" + aps() + "b");
    let expected = ["a", "b"];
    assert actual == expected;
}

#[test]
fn split3() {
    let actual = fs::split(ps() + "a" + ps() + "b");
    let expected = ["a", "b"];
    assert actual == expected;
}

#[test]
fn split4() {
    let actual = fs::split("a" + ps() + "b" + aps() + "c");
    let expected = ["a", "b", "c"];
    assert actual == expected;
}

#[test]
fn normalize1() {
    let actual = fs::normalize("a/b/..");
    let expected = "a";
    assert actual == expected;
}

#[test]
fn normalize2() {
    let actual = fs::normalize("/a/b/..");
    let expected = "/a";
    assert actual == expected;
}

#[test]
fn normalize3() {
    let actual = fs::normalize("a/../b");
    let expected = "b";
    assert actual == expected;
}

#[test]
fn normalize4() {
    let actual = fs::normalize("/a/../b");
    let expected = "/b";
    assert actual == expected;
}

#[test]
fn normalize5() {
    let actual = fs::normalize("a/.");
    let expected = "a";
    assert actual == expected;
}

#[test]
fn normalize6() {
    let actual = fs::normalize("a/./b/");
    let expected = "a/b/";
    assert actual == expected;
}

#[test]
fn normalize7() {
    let actual = fs::normalize("a/..");
    let expected = ".";
    assert actual == expected;
}

#[test]
fn normalize8() {
    let actual = fs::normalize("../../..");
    let expected = "../../..";
    assert actual == expected;
}

#[test]
fn normalize9() {
    let actual = fs::normalize("a/b/../../..");
    let expected = "..";
    assert actual == expected;
}

#[test]
fn normalize10() {
    let actual = fs::normalize("/a/b/c/../d/./../../e/");
    let expected = "/a/e/";
    log(error, actual);
    assert actual == expected;
}

#[test]
fn normalize11() {
    let actual = fs::normalize("/a/..");
    let expected = "/";
    assert actual == expected;
}

#[test]
#[cfg(target_os = "win32")]
fn normalize12() {
   let actual = fs::normalize("C:/whatever");
   let expected = "C:/whatever";
   log(error, actual);
   assert actual == expected;
}

#[test]
#[cfg(target_os = "win32")]
fn path_is_absolute_win32() {
   assert fs::path_is_absolute("C:/whatever");
}

#[test]
fn splitext_empty() {
    let (base, ext) = fs::splitext("");
    assert base == "";
    assert ext == "";
}

#[test]
fn splitext_ext() {
    let (base, ext) = fs::splitext("grum.exe");
    assert base == "grum";
    assert ext == ".exe";
}

#[test]
fn splitext_noext() {
    let (base, ext) = fs::splitext("grum");
    assert base == "grum";
    assert ext == "";
}

#[test]
fn splitext_dotfile() {
    let (base, ext) = fs::splitext(".grum");
    assert base == ".grum";
    assert ext == "";
}

#[test]
fn splitext_path_ext() {
    let (base, ext) = fs::splitext("oh/grum.exe");
    assert base == "oh/grum";
    assert ext == ".exe";
}

#[test]
fn splitext_path_noext() {
    let (base, ext) = fs::splitext("oh/grum");
    assert base == "oh/grum";
    assert ext == "";
}

#[test]
fn splitext_dot_in_path() {
    let (base, ext) = fs::splitext("oh.my/grum");
    assert base == "oh.my/grum";
    assert ext == "";
}

#[test]
fn splitext_nobasename() {
    let (base, ext) = fs::splitext("oh.my/");
    assert base == "oh.my/";
    assert ext == "";
}
