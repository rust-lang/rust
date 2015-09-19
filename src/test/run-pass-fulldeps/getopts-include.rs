// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test
// ignore-pretty: does not work well with `--test`

#![feature(rustc_private)]

extern crate getopts;
extern crate rustc_back;

use std::char;
use std::fs::File;
use std::io::{self, Write};

use getopts::*;
use getopts::Fail::*;

use rustc_back::tempdir::TempDir;

fn create_tmp_include(filename: &str, content: &str) -> (TempDir, String) {
    let tmp = TempDir::new("getopts-test").unwrap();

    let path = tmp.path().join(filename);
    let mut file = File::create(&path).unwrap();
    file.write_all(content.as_bytes()).unwrap();

    (tmp, path.to_str().unwrap().to_owned())
}

#[test]
fn test_include_no_recursive() {
    let (_tmp, path) = create_tmp_include("with-include.tmp", "--inc\0dummy.tmp");

    let args = vec!["--inc".to_owned(), path];
    let optgroups = vec![optinclude("", "inc", "desc", "FILE", b'\0')];

    match getopts(&args, &optgroups) {
        Err(NestedIncludeNotAllowed) => (),
        r => panic!("Expected Err(NestedIncludeNotAllowed), got: {:?}", r),
    }
}

#[test]
fn test_include_non_existent() {
    let args = vec!["--inc".to_owned(), "non-existent-include.tmp".to_owned()];
    let optgroups = vec![optinclude("", "inc", "desc", "FILE", b'\0')];
    match getopts(&args, &optgroups) {
        Err(GenericError(ref err)) if err.is::<io::Error>() => (),
        r => panic!("Expected Err(GenericError(io::Error)), got: {:?}", r),
    }
}

#[test]
#[should_panic(expected="hasarg == Yes")]
fn test_include_requires_hasarg_yes() {
    let optgroups = vec![opt("", "inc", "desc", "FILE", HasArg::No,
                         Occur::Multi, Special::Include(b'\0'))];
    // ignore the result because getopts should panic
    let _ = getopts(&vec![], &optgroups);
}

#[test]
fn test_include_accross_boundary() {
    let (_tmp, path) = create_tmp_include("boundary.tmp", "--with-argument");

    let args = vec!["--inc".to_owned(), path, "argument".to_owned()];
    let optgroups = vec![optinclude("", "inc", "desc", "FILE", b'\0'),
                         reqopt("", "with-argument", "desc", "ARG")];

    match getopts(&args, &optgroups) {
        Err(ArgumentMissing(ref arg)) if arg == "with-argument" => (),
        r => panic!("Expected Err(ArgumentMissing(\"with-argument\")), got: {:?}", r),
    }
}

#[test]
fn test_include_scoped_free_separator() {
    let (_tmp, path) = create_tmp_include("free.tmp", "--");

    let args = vec!["--inc".to_owned(), path, "--arg".to_owned()];
    let optgroups = vec![optinclude("", "inc", "desc", "FILE", b'\0'),
                         optflag("", "arg", "desc")];

    let matches = getopts(&args, &optgroups).unwrap();
    assert!(matches.opt_present("arg"));
}

#[test]
fn test_include_present_in_matches() {
    let (_tmp, path) = create_tmp_include("free.tmp", "--");

    let args = vec!["--inc".to_owned(), path.clone()];
    let optgroups = vec![optinclude("", "inc", "desc", "FILE", b'\0')];

    let matches = getopts(&args, &optgroups).unwrap();
    assert_eq!(matches.opt_strs("inc"), &[path]);
}

#[test]
fn test_include_special_chars() {
    let chars: String = (1..512).map(|i| char::from_u32(i).unwrap()).collect();

    let mut inc_args = String::with_capacity(511 * 2 + 5 * 2 + 3);
    inc_args.push_str("--arg");
    inc_args.push('\0');
    inc_args.push_str(&chars);
    inc_args.push('\0');
    inc_args.push_str("--arg");
    inc_args.push('\0');
    inc_args.push_str(&chars);

    let (_tmp, path) = create_tmp_include("free.tmp", &inc_args);

    let args = vec!["--inc".to_owned(), path];
    let optgroups = vec![optinclude("", "inc", "desc", "FILE", b'\0'),
                         optmulti("", "arg", "desc", "ARG")];

    let matches = getopts(&args, &optgroups).unwrap();
    assert_eq!(matches.opt_strs("arg"), &[chars.clone(), chars]);
}
