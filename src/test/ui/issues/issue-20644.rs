// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(stable_features)]

// A reduced version of the rustbook ice. The problem this encountered
// had to do with codegen ignoring binders.

// pretty-expanded FIXME #23616
// ignore-cloudabi no std::fs

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
