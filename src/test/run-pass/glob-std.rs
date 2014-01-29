// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast check-fast doesn't like 'extern mod extra'
// xfail-win32 TempDir may cause IoError on windows: #10462

extern mod extra;
extern mod glob;

use glob::glob;
use extra::tempfile::TempDir;
use std::unstable::finally::Finally;
use std::{os, unstable};
use std::io;

pub fn main() {
    fn mk_file(path: &str, directory: bool) {
        if directory {
            io::fs::mkdir(&Path::new(path), io::UserRWX);
        } else {
            io::File::create(&Path::new(path));
        }
    }

    fn abs_path(path: &str) -> Path {
        os::getcwd().join(&Path::new(path))
    }

    fn glob_vec(pattern: &str) -> ~[Path] {
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
    if os::consts::FAMILY != os::consts::windows::FAMILY {
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

    assert_eq!(glob_vec(""), ~[]);
    assert_eq!(glob_vec("."), ~[]);
    assert_eq!(glob_vec(".."), ~[]);

    assert_eq!(glob_vec("aaa"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("aaa/"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("a"), ~[]);
    assert_eq!(glob_vec("aa"), ~[]);
    assert_eq!(glob_vec("aaaa"), ~[]);

    assert_eq!(glob_vec("aaa/apple"), ~[abs_path("aaa/apple")]);
    assert_eq!(glob_vec("aaa/apple/nope"), ~[]);

    // windows should support both / and \ as directory separators
    if os::consts::FAMILY == os::consts::windows::FAMILY {
        assert_eq!(glob_vec("aaa\\apple"), ~[abs_path("aaa/apple")]);
    }

    assert_eq!(glob_vec("???/"), ~[
        abs_path("aaa"),
        abs_path("bbb"),
        abs_path("ccc"),
        abs_path("xyz")]);

    assert_eq!(glob_vec("aaa/tomato/tom?to.txt"), ~[
        abs_path("aaa/tomato/tomato.txt"),
        abs_path("aaa/tomato/tomoto.txt")]);

    assert_eq!(glob_vec("xyz/?"), ~[
        abs_path("xyz/x"),
        abs_path("xyz/y"),
        abs_path("xyz/z")]);

    assert_eq!(glob_vec("a*"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("*a*"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("a*a"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("aaa*"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("*aaa"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("*aaa*"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("*a*a*a*"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("aaa*/"), ~[abs_path("aaa")]);

    assert_eq!(glob_vec("aaa/*"), ~[
        abs_path("aaa/apple"),
        abs_path("aaa/orange"),
        abs_path("aaa/tomato")]);

    assert_eq!(glob_vec("aaa/*a*"), ~[
        abs_path("aaa/apple"),
        abs_path("aaa/orange"),
        abs_path("aaa/tomato")]);

    assert_eq!(glob_vec("*/*/*.txt"), ~[
        abs_path("aaa/tomato/tomato.txt"),
        abs_path("aaa/tomato/tomoto.txt")]);

    assert_eq!(glob_vec("*/*/t[aob]m?to[.]t[!y]t"), ~[
        abs_path("aaa/tomato/tomato.txt"),
        abs_path("aaa/tomato/tomoto.txt")]);

    assert_eq!(glob_vec("aa[a]"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("aa[abc]"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("a[bca]a"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("aa[b]"), ~[]);
    assert_eq!(glob_vec("aa[xyz]"), ~[]);
    assert_eq!(glob_vec("aa[]]"), ~[]);

    assert_eq!(glob_vec("aa[!b]"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("aa[!bcd]"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("a[!bcd]a"), ~[abs_path("aaa")]);
    assert_eq!(glob_vec("aa[!a]"), ~[]);
    assert_eq!(glob_vec("aa[!abc]"), ~[]);

    assert_eq!(glob_vec("bbb/specials/[[]"), ~[abs_path("bbb/specials/[")]);
    assert_eq!(glob_vec("bbb/specials/!"), ~[abs_path("bbb/specials/!")]);
    assert_eq!(glob_vec("bbb/specials/[]]"), ~[abs_path("bbb/specials/]")]);

    if os::consts::FAMILY != os::consts::windows::FAMILY {
        assert_eq!(glob_vec("bbb/specials/[*]"), ~[abs_path("bbb/specials/*")]);
        assert_eq!(glob_vec("bbb/specials/[?]"), ~[abs_path("bbb/specials/?")]);
    }

    if os::consts::FAMILY == os::consts::windows::FAMILY {

        assert_eq!(glob_vec("bbb/specials/[![]"), ~[
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/]")]);

        assert_eq!(glob_vec("bbb/specials/[!]]"), ~[
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/[")]);

        assert_eq!(glob_vec("bbb/specials/[!!]"), ~[
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")]);

    } else {

        assert_eq!(glob_vec("bbb/specials/[![]"), ~[
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/]")]);

        assert_eq!(glob_vec("bbb/specials/[!]]"), ~[
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/[")]);

        assert_eq!(glob_vec("bbb/specials/[!!]"), ~[
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")]);

        assert_eq!(glob_vec("bbb/specials/[!*]"), ~[
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/?"),
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")]);

        assert_eq!(glob_vec("bbb/specials/[!?]"), ~[
            abs_path("bbb/specials/!"),
            abs_path("bbb/specials/*"),
            abs_path("bbb/specials/["),
            abs_path("bbb/specials/]")]);

    }
}
