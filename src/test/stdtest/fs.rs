
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
