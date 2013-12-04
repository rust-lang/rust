// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast check-fast doesn't like 'extern mod'
// xfail-win32 TempDir may cause IoError on windows: #10463

// These tests are here to exercise the functionality of the `tempfile` module.
// One might expect these tests to be located in that module, but sadly they
// cannot. The tests need to invoke `os::change_dir` which cannot be done in the
// normal test infrastructure. If the tests change the current working
// directory, then *all* tests which require relative paths suddenly break b/c
// they're in a different location than before. Hence, these tests are all run
// serially here.

extern mod extra;

use extra::tempfile::TempDir;
use std::io::fs;
use std::io;
use std::os;
use std::task;

fn test_tempdir() {
    let path = {
        let p = TempDir::new_in(&Path::new("."), "foobar").unwrap();
        let p = p.path();
        assert!(p.as_vec().ends_with(bytes!("foobar")));
        p.clone()
    };
    assert!(!path.exists());
}

fn test_rm_tempdir() {
    let (rd, wr) = stream();
    let f: proc() = proc() {
        let tmp = TempDir::new("test_rm_tempdir").unwrap();
        wr.send(tmp.path().clone());
        fail!("fail to unwind past `tmp`");
    };
    task::try(f);
    let path = rd.recv();
    assert!(!path.exists());

    let tmp = TempDir::new("test_rm_tempdir").unwrap();
    let path = tmp.path().clone();
    let f: proc() = proc() {
        let _tmp = tmp;
        fail!("fail to unwind past `tmp`");
    };
    task::try(f);
    assert!(!path.exists());

    let path;
    {
        let f: proc() -> TempDir = proc() {
            TempDir::new("test_rm_tempdir").unwrap()
        };
        let tmp = task::try(f).ok().expect("test_rm_tmdir");
        path = tmp.path().clone();
        assert!(path.exists());
    }
    assert!(!path.exists());

    let path;
    {
        let tmp = TempDir::new("test_rm_tempdir").unwrap();
        path = tmp.unwrap();
    }
    assert!(path.exists());
    fs::rmdir_recursive(&path);
    assert!(!path.exists());
}

// Ideally these would be in std::os but then core would need
// to depend on std
fn recursive_mkdir_rel() {
    let path = Path::new("frob");
    let cwd = os::getcwd();
    debug!("recursive_mkdir_rel: Making: {} in cwd {} [{:?}]", path.display(),
           cwd.display(), path.exists());
    fs::mkdir_recursive(&path, io::UserRWX);
    assert!(path.is_dir());
    fs::mkdir_recursive(&path, io::UserRWX);
    assert!(path.is_dir());
}

fn recursive_mkdir_dot() {
    let dot = Path::new(".");
    fs::mkdir_recursive(&dot, io::UserRWX);
    let dotdot = Path::new("..");
    fs::mkdir_recursive(&dotdot, io::UserRWX);
}

fn recursive_mkdir_rel_2() {
    let path = Path::new("./frob/baz");
    let cwd = os::getcwd();
    debug!("recursive_mkdir_rel_2: Making: {} in cwd {} [{:?}]", path.display(),
           cwd.display(), path.exists());
    fs::mkdir_recursive(&path, io::UserRWX);
    assert!(path.is_dir());
    assert!(path.dir_path().is_dir());
    let path2 = Path::new("quux/blat");
    debug!("recursive_mkdir_rel_2: Making: {} in cwd {}", path2.display(),
           cwd.display());
    fs::mkdir_recursive(&path2, io::UserRWX);
    assert!(path2.is_dir());
    assert!(path2.dir_path().is_dir());
}

// Ideally this would be in core, but needs TempFile
pub fn test_rmdir_recursive_ok() {
    let rwx = io::UserRWX;

    let tmpdir = TempDir::new("test").expect("test_rmdir_recursive_ok: \
                                              couldn't create temp dir");
    let tmpdir = tmpdir.path();
    let root = tmpdir.join("foo");

    debug!("making {}", root.display());
    fs::mkdir(&root, rwx);
    fs::mkdir(&root.join("foo"), rwx);
    fs::mkdir(&root.join("foo").join("bar"), rwx);
    fs::mkdir(&root.join("foo").join("bar").join("blat"), rwx);
    fs::rmdir_recursive(&root);
    assert!(!root.exists());
    assert!(!root.join("bar").exists());
    assert!(!root.join("bar").join("blat").exists());
}

fn in_tmpdir(f: ||) {
    let tmpdir = TempDir::new("test").expect("can't make tmpdir");
    assert!(os::change_dir(tmpdir.path()));

    f();
}

fn main() {
    in_tmpdir(test_tempdir);
    in_tmpdir(test_rm_tempdir);
    in_tmpdir(recursive_mkdir_rel);
    in_tmpdir(recursive_mkdir_dot);
    in_tmpdir(recursive_mkdir_rel_2);
    in_tmpdir(test_rmdir_recursive_ok);
}
