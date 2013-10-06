// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test can't be a unit test in std,
// because it needs mkdtemp, which is in extra

// xfail-fast
extern mod extra;

use extra::tempfile::mkdtemp;
use std::os;
use std::libc;

fn rename_directory() {
    #[fixed_stack_segment];
    unsafe {
        static U_RWX: i32 = (libc::S_IRUSR | libc::S_IWUSR | libc::S_IXUSR) as i32;

        let tmpdir = mkdtemp(&os::tmpdir(), "rename_directory").expect("rename_directory failed");
        let old_path = tmpdir.push_many(["foo", "bar", "baz"]);
        assert!(os::mkdir_recursive(&old_path, U_RWX));
        let test_file = &old_path.push("temp.txt");

        /* Write the temp input file */
        let ostream = do test_file.to_str().with_c_str |fromp| {
            do "w+b".with_c_str |modebuf| {
                libc::fopen(fromp, modebuf)
            }
        };
        assert!((ostream as uint != 0u));
        let s = ~"hello";
        do "hello".with_c_str |buf| {
            let write_len = libc::fwrite(buf as *libc::c_void,
                                         1u as libc::size_t,
                                         (s.len() + 1u) as libc::size_t,
                                         ostream);
            assert_eq!(write_len, (s.len() + 1) as libc::size_t)
        }
        assert_eq!(libc::fclose(ostream), (0u as libc::c_int));

        let new_path = tmpdir.push_many(["quux", "blat"]);
        assert!(os::mkdir_recursive(&new_path, U_RWX));
        assert!(os::rename_file(&old_path, &new_path.push("newdir")));
        assert!(os::path_is_dir(&new_path.push("newdir")));
        assert!(os::path_exists(&new_path.push_many(["newdir", "temp.txt"])));
    }
}

fn main() { rename_directory() }
