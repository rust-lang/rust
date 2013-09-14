// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast windows doesn't like 'extern mod extra'

// These tests are here to exercise the functionality of the `tempfile` module.
// One might expect these tests to be located in that module, but sadly they
// cannot. The tests need to invoke `os::change_dir` which cannot be done in the
// normal test infrastructure. If the tests change the current working
// directory, then *all* tests which require relative paths suddenly break b/c
// they're in a different location than before. Hence, these tests are all run
// serially here.

extern mod extra;

use extra::tempfile::mkdtemp;
use std::os;
use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};

fn test_mkdtemp() {
    let p = mkdtemp(&Path("."), "foobar").unwrap();
    os::remove_dir(&p);
    assert!(p.to_str().ends_with("foobar"));
}

// Ideally these would be in std::os but then core would need
// to depend on std
fn recursive_mkdir_rel() {
    let path = Path("frob");
    debug!("recursive_mkdir_rel: Making: %s in cwd %s [%?]", path.to_str(),
           os::getcwd().to_str(),
           os::path_exists(&path));
    assert!(os::mkdir_recursive(&path,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    assert!(os::path_is_dir(&path));
    assert!(os::mkdir_recursive(&path,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    assert!(os::path_is_dir(&path));
}

fn recursive_mkdir_dot() {
    let dot = Path(".");
    assert!(os::mkdir_recursive(&dot,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    let dotdot = Path("..");
    assert!(os::mkdir_recursive(&dotdot,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
}

fn recursive_mkdir_rel_2() {
    let path = Path("./frob/baz");
    debug!("recursive_mkdir_rel_2: Making: %s in cwd %s [%?]", path.to_str(),
           os::getcwd().to_str(), os::path_exists(&path));
    assert!(os::mkdir_recursive(&path, (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
        assert!(os::path_is_dir(&path));
    assert!(os::path_is_dir(&path.pop()));
    let path2 = Path("quux/blat");
    debug!("recursive_mkdir_rel_2: Making: %s in cwd %s", path2.to_str(),
           os::getcwd().to_str());
    assert!(os::mkdir_recursive(&path2, (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
        assert!(os::path_is_dir(&path2));
    assert!(os::path_is_dir(&path2.pop()));
}

// Ideally this would be in core, but needs mkdtemp
pub fn test_rmdir_recursive_ok() {
    let rwx = (S_IRUSR | S_IWUSR | S_IXUSR) as i32;

    let tmpdir = mkdtemp(&os::tmpdir(), "test").expect("test_rmdir_recursive_ok: \
                                        couldn't create temp dir");
    let root = tmpdir.push("foo");

    debug!("making %s", root.to_str());
    assert!(os::make_dir(&root, rwx));
    assert!(os::make_dir(&root.push("foo"), rwx));
    assert!(os::make_dir(&root.push("foo").push("bar"), rwx));
    assert!(os::make_dir(&root.push("foo").push("bar").push("blat"), rwx));
    assert!(os::remove_dir_recursive(&root));
    assert!(!os::path_exists(&root));
    assert!(!os::path_exists(&root.push("bar")));
    assert!(!os::path_exists(&root.push("bar").push("blat")));
}

fn in_tmpdir(f: &fn()) {
    let tmpdir = mkdtemp(&os::tmpdir(), "test").expect("can't make tmpdir");
    assert!(os::change_dir(&tmpdir));

    f();
}

fn main() {
    in_tmpdir(test_mkdtemp);
    in_tmpdir(recursive_mkdir_rel);
    in_tmpdir(recursive_mkdir_dot);
    in_tmpdir(recursive_mkdir_rel_2);
    in_tmpdir(test_rmdir_recursive_ok);
}
