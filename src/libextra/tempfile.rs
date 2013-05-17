// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Temporary files and directories

use core::prelude::*;

use core::rand::RngUtil;

pub fn mkdtemp(tmpdir: &Path, suffix: &str) -> Option<Path> {
    let mut r = rand::rng();
    for 1000.times {
        let p = tmpdir.push(r.gen_str(16) + suffix);
        if os::make_dir(&p, 0x1c0) { // 700
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use tempfile::mkdtemp;
    use tempfile;
    use core::os;

    #[test]
    fn test_mkdtemp() {
        let p = mkdtemp(&Path("."), "foobar").unwrap();
        os::remove_dir(&p);
        assert!(str::ends_with(p.to_str(), "foobar"));
    }

    // Ideally these would be in core::os but then core would need
    // to depend on std
    #[test]
    fn recursive_mkdir_rel() {
        use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
        use core::os;

        let root = mkdtemp(&os::tmpdir(), "recursive_mkdir_rel").
            expect("recursive_mkdir_rel");
        assert!(do os::change_dir_locked(&root) {
            let path = Path("frob");
            debug!("recursive_mkdir_rel: Making: %s in cwd %s [%?]", path.to_str(),
                   os::getcwd().to_str(),
                   os::path_exists(&path));
            assert!(os::mkdir_recursive(&path,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
            assert!(os::path_is_dir(&path));
            assert!(os::mkdir_recursive(&path,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
            assert!(os::path_is_dir(&path));
        });
    }

    #[test]
    fn recursive_mkdir_dot() {
        use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
        use core::os;

        let dot = Path(".");
        assert!(os::mkdir_recursive(&dot,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
        let dotdot = Path("..");
        assert!(os::mkdir_recursive(&dotdot,  (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    }

    #[test]
    fn recursive_mkdir_rel_2() {
        use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
        use core::os;

        let root = mkdtemp(&os::tmpdir(), "recursive_mkdir_rel_2").
            expect("recursive_mkdir_rel_2");
        assert!(do os::change_dir_locked(&root) {
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
        });
    }

    // Ideally this would be in core, but needs mkdtemp
    #[test]
    pub fn test_rmdir_recursive_ok() {
        use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
        use core::os;

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
}
