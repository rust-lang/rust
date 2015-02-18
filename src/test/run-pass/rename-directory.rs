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

extern crate libc;

use std::ffi::CString;
use std::old_io::TempDir;
use std::old_io::fs::PathExtensions;
use std::old_io::fs;
use std::old_io;
use std::os;

fn rename_directory() {
    unsafe {
        static U_RWX: i32 = (libc::S_IRUSR | libc::S_IWUSR | libc::S_IXUSR) as i32;

        let tmpdir = TempDir::new("rename_directory").ok().expect("rename_directory failed");
        let tmpdir = tmpdir.path();
        let old_path = tmpdir.join_many(&["foo", "bar", "baz"]);
        fs::mkdir_recursive(&old_path, old_io::USER_RWX);
        let test_file = &old_path.join("temp.txt");

        /* Write the temp input file */
        let fromp = CString::new(test_file.as_vec()).unwrap();
        let modebuf = CString::new(b"w+b").unwrap();
        let ostream = libc::fopen(fromp.as_ptr(), modebuf.as_ptr());
        assert!((ostream as uint != 0_usize));
        let s = "hello".to_string();
        let buf = CString::new(b"hello").unwrap();
        let write_len = libc::fwrite(buf.as_ptr() as *mut _,
                                     1_usize as libc::size_t,
                                     (s.len() + 1_usize) as libc::size_t,
                                     ostream);
        assert_eq!(write_len, (s.len() + 1) as libc::size_t);
        assert_eq!(libc::fclose(ostream), (0_usize as libc::c_int));

        let new_path = tmpdir.join_many(&["quux", "blat"]);
        fs::mkdir_recursive(&new_path, old_io::USER_RWX);
        fs::rename(&old_path, &new_path.join("newdir"));
        assert!(new_path.join("newdir").is_dir());
        assert!(new_path.join_many(&["newdir", "temp.txt"]).exists());
    }
}

pub fn main() { rename_directory() }
