// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows TempDir may cause IoError on windows: #10462

#![feature(macro_rules)]

extern crate glob;

use glob::glob;
use std::os;
use std::io;
use std::io::TempDir;

macro_rules! assert_eq ( ($e1:expr, $e2:expr) => (
    if $e1 != $e2 {
        fail!("{} != {}", stringify!($e1), stringify!($e2))
    }
) )

pub fn main() {
    fn mk_file(path: &str, directory: bool) {
        if directory {
            io::fs::mkdir(&Path::new(path), io::UserRWX).unwrap();
        } else {
            io::File::create(&Path::new(path)).unwrap();
        }
    }

    fn abs_path(path: &str) -> Path {
        os::getcwd().join(&Path::new(path))
    }

    fn glob_vec(pattern: &str) -> Vec<Path> {
        glob(pattern).collect()
    }

    let root = TempDir::new("glob-tests");
    let root = root.expect("Should have created a temp directory");
    assert!(os::change_dir(root.path()));

    mk_file("aaa", true);
    mk_file("aaa/apple", true);
    mk_file("aaa/orange", true);
    mk_file("aaa/tomato", true);
    mk_file("aaa/tomato/tomato.txt", false);
    mk_file("aaa/tomato/tomoto.txt", false);
    mk_file("bbb", true);
    mk_file("bbb/specials", true);
    mk_file("bbb/specials/!", false);

    // windows does not allow `*` or `?` characters to exist in filenames
    if os::consts::FAMILY != "windows" {
        mk_file("bbb/specials/*", false);
        mk_file("bbb/specials/?", false);
    }

    mk_file("bbb/specials/[", false);
    mk_file("bbb/specials/]", false);
    mk_file("ccc", true);
    mk_file("xyz", true);
    mk_file("xyz/x", false);
    mk_file("xyz/y", false);
    mk_file("xyz/z", false);

    assert_eq!(glob_vec(""), Vec::new());
    assert_eq!(glob_vec("."), vec!(os::getcwd()));
    assert_eq!(glob_vec(".."), vec!(os::getcwd().join("..")));

    assert_eq!(glob_vec("aaa"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("aaa/"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("a"), Vec::new());
    assert_eq!(glob_vec("aa"), Vec::new());
    assert_eq!(glob_vec("aaaa"), Vec::new());

    assert_eq!(glob_vec("aaa/apple"), vec!(abs_path("aaa/apple")));
    assert_eq!(glob_vec("aaa/apple/nope"), Vec::new());

    // windows should support both / and \ as directory separators
    if os::consts::FAMILY == "windows" {
        assert_eq!(glob_vec("aaa\\apple"), vec!(abs_path("aaa/apple")));
    }

    assert_eq!(glob_vec("???/"), vec!(
        abs_path("aaa"),
        abs_path("bbb"),
        abs_path("ccc"),
        abs_path("xyz")));

    assert_eq!(glob_vec("aaa/tomato/tom?to.txt"), vec!(
        abs_path("aaa/tomato/tomato.txt"),
        abs_path("aaa/tomato/tomoto.txt")));

    assert_eq!(glob_vec("xyz/?"), vec!(
        abs_path("xyz/x"),
        abs_path("xyz/y"),
        abs_path("xyz/z")));

    assert_eq!(glob_vec("a*"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("*a*"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("a*a"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("aaa*"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("*aaa"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("*aaa*"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("*a*a*a*"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("aaa*/"), vec!(abs_path("aaa")));

    assert_eq!(glob_vec("aaa/*"), vec!(
        abs_path("aaa/apple"),
        abs_path("aaa/orange"),
        abs_path("aaa/tomato")));

    assert_eq!(glob_vec("aaa/*a*"), vec!(
        abs_path("aaa/apple"),
        abs_path("aaa/orange"),
        abs_path("aaa/tomato")));

    assert_eq!(glob_vec("*/*/*.txt"), vec!(
        abs_path("aaa/tomato/tomato.txt"),
        abs_path("aaa/tomato/tomoto.txt")));

    assert_eq!(glob_vec("*/*/t[aob]m?to[.]t[!y]t"), vec!(
        abs_path("aaa/tomato/tomato.txt"),
        abs_path("aaa/tomato/tomoto.txt")));

    assert_eq!(glob_vec("./aaa"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("./*"), glob_vec("*"));
    assert_eq!(glob_vec("*/..").pop().unwrap(), abs_path("."));
    assert_eq!(glob_vec("aaa/../bbb"), vec!(abs_path("bbb")));
    assert_eq!(glob_vec("nonexistent/../bbb"), Vec::new());
    assert_eq!(glob_vec("aaa/tomato/tomato.txt/.."), Vec::new());

    assert_eq!(glob_vec("aaa/tomato/tomato.txt/"), Vec::new());

    assert_eq!(glob_vec("aa[a]"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("aa[abc]"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("a[bca]a"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("aa[b]"), Vec::new());
    assert_eq!(glob_vec("aa[xyz]"), Vec::new());
    assert_eq!(glob_vec("aa[]]"), Vec::new());

    assert_eq!(glob_vec("aa[!b]"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("aa[!bcd]"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("a[!bcd]a"), vec!(abs_path("aaa")));
    assert_eq!(glob_vec("aa[!a]"), Vec::new());
    assert_eq!(glob_vec("aa[!abc]"), Vec::new());

    assert_eq!(glob_vec("bbb/specials/[[]"), vec!(abs_path("bbb/specials/[")));
    assert_eq!(glob_vec("bbb/specials/!"), vec!(abs_path("bbb/specials/!")));
    assert_eq!(glob_vec("bbb/specials/[]]"), vec!(abs_path("bbb/specials/]")));

    if os::consts::FAMILY != "windows" {
        assert_eq!(glob_vec("bbb/specials/[*]"), vec!(abs_path("bbb/specials/*")));
        assert_eq!(glob_vec("bbb/specials/[?]"), vec!(abs_path("bbb/specials/?")));
    }

    if os::consts::FAMILY == "windows" {

        assert_eq!(glob_vec("bbb/specials/[![]"), vec!(
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/]")));

        assert_eq!(glob_vec("bbb/specials/[!]]"), vec!(
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/[")));

        assert_eq!(glob_vec("bbb/specials/[!!]"), vec!(
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")));

    } else {

        assert_eq!(glob_vec("bbb/specials/[![]"), vec!(
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/]")));

        assert_eq!(glob_vec("bbb/specials/[!]]"), vec!(
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/[")));

        assert_eq!(glob_vec("bbb/specials/[!!]"), vec!(
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")));

        assert_eq!(glob_vec("bbb/specials/[!*]"), vec!(
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")));

        assert_eq!(glob_vec("bbb/specials/[!?]"), vec!(
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")));

    }
}
