//@ run-pass
// Testing that a librustc_ast can parse modules with canonicalized base path
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use rustc_parse::{lexer::StripTokens, new_parser_from_file, unwrap_or_emit_fatal};
use rustc_session::parse::ParseSess;
use std::path::Path;

#[path = "mod_dir_simple/test.rs"]
mod gravy;

pub fn main() {
    rustc_span::create_default_session_globals_then(|| parse());

    assert_eq!(gravy::foo(), 10);
}

fn parse() {
    let psess = ParseSess::new(vec![rustc_parse::DEFAULT_LOCALE_RESOURCE]);

    let path = Path::new(file!());
    let path = path.canonicalize().unwrap();
    let mut parser = unwrap_or_emit_fatal(new_parser_from_file(
        &psess,
        &path,
        StripTokens::ShebangAndFrontmatter,
        None,
    ));
    let _ = parser.parse_crate_mod();
}
