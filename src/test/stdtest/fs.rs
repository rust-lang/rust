
use std;
import std::fs;

#[test]
fn test_connect() {
    let slash = fs::path_sep();
    log_err fs::connect("a", "b");
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
    assert (std::vec::len(dirs) > 0u);

    for dir in dirs { log dir; }
}

#[test]
fn file_is_dir() {
    assert (fs::file_is_dir("."));
    assert (!fs::file_is_dir("test/stdtest/fs.rs"));
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
    assert actual == expected;
}

#[test]
fn normalize11() {
    let actual = fs::normalize("/a/..");
    let expected = "/";
    assert actual == expected;
}