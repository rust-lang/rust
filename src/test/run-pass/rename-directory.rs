// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test can't be a unit test in std,
// because it needs TempDir, which is in extra

// ignore-fast

use std::io::TempDir;
use std::os;
use std::libc;
use std::io;
use std::io::fs;

fn rename_directory() {
    unsafe {
        static U_RWX: i32 = (libc::S_IRUSR | libc::S_IWUSR | libc::S_IXUSR) as i32;

        let tmpdir = TempDir::new("rename_directory").expect("rename_directory failed");
        let tmpdir = tmpdir.path();
        let old_path = tmpdir.join_many(["foo", "bar", "baz"]);
        fs::mkdir_recursive(&old_path, io::UserRWX);
        let test_file = &old_path.join("temp.txt");

        /* Write the temp input file */
        let ostream = test_file.with_c_str(|fromp| {
            "w+b".with_c_str(|modebuf| {
                libc::fopen(fromp, modebuf)
            })
        });
        assert!((ostream as uint != 0u));
        let s = ~"hello";
        "hello".with_c_str(|buf| {
            let write_len = libc::fwrite(buf as *libc::c_void,
                                         1u as libc::size_t,
                                         (s.len() + 1u) as libc::size_t,
                                         ostream);
            assert_eq!(write_len, (s.len() + 1) as libc::size_t)
        });
        assert_eq!(libc::fclose(ostream), (0u as libc::c_int));

        let new_path = tmpdir.join_many(["quux", "blat"]);
        fs::mkdir_recursive(&new_path, io::UserRWX);
        fs::rename(&old_path, &new_path.join("newdir"));
        assert!(new_path.join("newdir").is_dir());
        assert!(new_path.join_many(["newdir", "temp.txt"]).exists());
    }
}

pub fn main() { rename_directory() }
