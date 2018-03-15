// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fs;
use std::io;

fn assert_invalid_input<T>(on: &str, result: io::Result<T>) {
    fn inner(on: &str, result: io::Result<()>) {
        match result {
            Ok(()) => panic!("{} didn't return an error on a path with NUL", on),
            Err(e) => assert!(e.kind() == io::ErrorKind::InvalidInput,
                              "{} returned a strange {:?} on a path with NUL", on, e.kind()),
        }
    }
    inner(on, result.map(|_| ()))
}

fn main() {
    assert_invalid_input("File::open", fs::File::open("\0"));
    assert_invalid_input("File::create", fs::File::create("\0"));
    assert_invalid_input("remove_file", fs::remove_file("\0"));
    assert_invalid_input("metadata", fs::metadata("\0"));
    assert_invalid_input("symlink_metadata", fs::symlink_metadata("\0"));
    assert_invalid_input("rename1", fs::rename("\0", "a"));
    assert_invalid_input("rename2", fs::rename("a", "\0"));
    assert_invalid_input("copy1", fs::copy("\0", "a"));
    assert_invalid_input("copy2", fs::copy("a", "\0"));
    assert_invalid_input("hard_link1", fs::hard_link("\0", "a"));
    assert_invalid_input("hard_link2", fs::hard_link("a", "\0"));
    assert_invalid_input("soft_link1", fs::soft_link("\0", "a"));
    assert_invalid_input("soft_link2", fs::soft_link("a", "\0"));
    assert_invalid_input("read_link", fs::read_link("\0"));
    assert_invalid_input("canonicalize", fs::canonicalize("\0"));
    assert_invalid_input("create_dir", fs::create_dir("\0"));
    assert_invalid_input("create_dir_all", fs::create_dir_all("\0"));
    assert_invalid_input("remove_dir", fs::remove_dir("\0"));
    assert_invalid_input("remove_dir_all", fs::remove_dir_all("\0"));
    assert_invalid_input("read_dir", fs::read_dir("\0"));
    assert_invalid_input("set_permissions",
                         fs::set_permissions("\0", fs::metadata(".").unwrap().permissions()));
}
