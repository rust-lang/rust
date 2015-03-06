// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A reduced version of the rustbook ice. The problem this encountered
// had to do with trans ignoring binders.

#![feature(os)]

use std::iter;
use std::os;
use std::fs::File;
use std::io::prelude::*;
use std::env;
use std::path::Path;

pub fn parse_summary<R: Read>(_: R, _: &Path) {
     let path_from_root = Path::new("");
     Path::new(&iter::repeat("../")
               .take(path_from_root.components().count() - 1)
               .collect::<String>());
 }

fn foo() {
    let cwd = env::current_dir().unwrap();
    let src = cwd.clone();
    let summary = File::open(&src.join("SUMMARY.md")).unwrap();
    let _ = parse_summary(summary, &src);
}

fn main() {}
