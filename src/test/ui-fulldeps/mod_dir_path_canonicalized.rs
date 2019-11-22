// run-pass
// Testing that a libsyntax can parse modules with canonicalized base path
// ignore-cross-compile

#![feature(rustc_private)]

extern crate syntax;
extern crate syntax_expand;
extern crate rustc_parse;

use rustc_parse::new_parser_from_file;
use std::path::Path;
use syntax::sess::ParseSess;
use syntax::source_map::FilePathMapping;

#[path = "mod_dir_simple/test.rs"]
mod gravy;

pub fn main() {
    syntax::with_default_globals(|| parse());

    assert_eq!(gravy::foo(), 10);
}

fn parse() {
    let parse_session = ParseSess::new(FilePathMapping::empty());

    let path = Path::new(file!());
    let path = path.canonicalize().unwrap();
    let mut parser = new_parser_from_file(&parse_session, &path);
    let _ = parser.parse_crate_mod();
}
