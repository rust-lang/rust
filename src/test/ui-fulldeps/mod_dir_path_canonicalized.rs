// run-pass
// Testing that a librustc_ast can parse modules with canonicalized base path
// ignore-cross-compile
// ignore-remote

#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;

use rustc_parse::new_parser_from_file;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::FilePathMapping;
use std::path::Path;

#[path = "mod_dir_simple/test.rs"]
mod gravy;

pub fn main() {
    rustc_ast::with_default_globals(|| parse());

    assert_eq!(gravy::foo(), 10);
}

fn parse() {
    let parse_session = ParseSess::new(FilePathMapping::empty());

    let path = Path::new(file!());
    let path = path.canonicalize().unwrap();
    let mut parser = new_parser_from_file(&parse_session, &path, None);
    let _ = parser.parse_crate_mod();
}
