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

use extra::tempfile::TempDir;
use std::os;
use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use std::task;
use std::cell::Cell;

fn test_tempdir() {
    let path = {
        let p = TempDir::new_in(&Path::new("."), "foobar").unwrap();
        let p = p.path();
        assert!(ends_with(p.as_vec(), bytes!("foobar")));
        p.clone()
    };
    assert!(!os::path_exists(&path));
    fn ends_with(v: &[u8], needle: &[u8]) -> bool {
        v.len() >= needle.len() && v.slice_from(v.len()-needle.len()) == needle
    }
}

fn test_rm_tempdir() {
    let (rd, wr) = stream();
    let f: ~fn() = || {
        let tmp = TempDir::new("test_rm_tempdir").unwrap();
        wr.send(tmp.path().clone());
        fail2!("fail to unwind past `tmp`");
    };
    task::try(f);
    let path = rd.recv();
    assert!(!os::path_exists(&path));

    let tmp = TempDir::new("test_rm_tempdir").unwrap();
    let path = tmp.path().clone();
    let cell = Cell::new(tmp);
    let f: ~fn() = || {
        let _tmp = cell.take();
        fail2!("fail to unwind past `tmp`");
    };
    task::try(f);
    assert!(!os::path_exists(&path));

    let path;
    {
        let f: ~fn() -> TempDir = || {
            TempDir::new("test_rm_tempdir").unwrap()
        };
        let tmp = task::try(f).expect("test_rm_tmdir");
        path = tmp.path().clone();
        assert!(os::path_exists(&path));
    }
    assert!(!os::path_exists(&path));

    let path;
    {
        let tmp = TempDir::new("test_rm_tempdir").unwrap();
        path = tmp.unwrap();
    }
    assert!(os::path_exists(&path));
    os::remove_dir_recursive(&path);
    assert!(!os::path_exists(&path));
}

// Ideally these would be in std::os but then core would need
// to depend on std
fn recursive_mkdir_rel() {
    let path = Path::new("frob");
    let cwd = os::getcwd();
    debug2!("recursive_mkdir_rel: Making: {} in cwd {} [{:?}]", path.display(),
           cwd.display(), os::path_exists(&path));
    assert!(os::mkdir_recursive(&path,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    assert!(os::path_is_dir(&path));
    assert!(os::mkdir_recursive(&path,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    assert!(os::path_is_dir(&path));
}

fn recursive_mkdir_dot() {
    let dot = Path::new(".");
    assert!(os::mkdir_recursive(&dot,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    let dotdot = Path::new("..");
    assert!(os::mkdir_recursive(&dotdot,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
}

fn recursive_mkdir_rel_2() {
    let path = Path::new("./frob/baz");
    let cwd = os::getcwd();
    debug2!("recursive_mkdir_rel_2: Making: {} in cwd {} [{:?}]", path.display(),
           cwd.display(), os::path_exists(&path));
    assert!(os::mkdir_recursive(&path, (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
        assert!(os::path_is_dir(&path));
    assert!(os::path_is_dir(&path.dir_path()));
    let path2 = Path::new("quux/blat");
    debug2!("recursive_mkdir_rel_2: Making: {} in cwd {}", path2.display(),
           cwd.display());
    assert!(os::mkdir_recursive(&path2, (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
        assert!(os::path_is_dir(&path2));
    assert!(os::path_is_dir(&path2.dir_path()));
}

// Ideally this would be in core, but needs TempFile
pub fn test_rmdir_recursive_ok() {
    let rwx = (S_IRUSR | S_IWUSR | S_IXUSR) as i32;

    let tmpdir = TempDir::new("test").expect("test_rmdir_recursive_ok: \
                                              couldn't create temp dir");
    let tmpdir = tmpdir.path();
    let root = tmpdir.join("foo");

    debug2!("making {}", root.display());
    assert!(os::make_dir(&root, rwx));
    assert!(os::make_dir(&root.join("foo"), rwx));
    assert!(os::make_dir(&root.join("foo").join("bar"), rwx));
    assert!(os::make_dir(&root.join("foo").join("bar").join("blat"), rwx));
    assert!(os::remove_dir_recursive(&root));
    assert!(!os::path_exists(&root));
    assert!(!os::path_exists(&root.join("bar")));
    assert!(!os::path_exists(&root.join("bar").join("blat")));
}

fn in_tmpdir(f: &fn()) {
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
