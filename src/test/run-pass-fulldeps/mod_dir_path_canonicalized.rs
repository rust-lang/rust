// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that a libsyntax can parse modules with canonicalized base path
// ignore-cross-compile

#![feature(rustc_private)]

extern crate syntax;

use std::path::Path;
use syntax::source_map::FilePathMapping;
use syntax::parse::{self, ParseSess};

#[path = "mod_dir_simple/test.rs"]
mod gravy;

pub fn main() {
    syntax::with_globals(|| parse());

    assert_eq!(gravy::foo(), 10);
}

fn parse() {
    let parse_session = ParseSess::new(FilePathMapping::empty());

    let path = Path::new(file!());
    let path = path.canonicalize().unwrap();
    let mut parser = parse::new_parser_from_file(&parse_session, &path);
    let _ = parser.parse_crate_mod();
}
