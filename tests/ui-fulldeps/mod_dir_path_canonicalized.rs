// run-pass
// Testing that a librustc_ast can parse modules with canonicalized base path
// ignore-cross-compile
// ignore-remote
// no-remap-src-base: Reading `file!()` (expectedly) fails when enabled.

#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use rustc_parse::new_parser_from_file;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::FilePathMapping;
use std::path::Path;

#[path = "mod_dir_simple/test.rs"]
mod gravy;

pub fn main() {
    rustc_span::create_default_session_globals_then(|| parse());

    assert_eq!(gravy::foo(), 10);
}

fn parse() {
    let parse_session = ParseSess::new(
        vec![rustc_parse::DEFAULT_LOCALE_RESOURCE],
        FilePathMapping::empty()
    );

    let path = Path::new(file!());
    let path = path.canonicalize().unwrap();
    let mut parser = new_parser_from_file(&parse_session, &path, None);
    let _ = parser.parse_crate_mod();
}
