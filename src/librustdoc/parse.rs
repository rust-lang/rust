//! AST-parsing helpers

use rustc::driver::driver;
use driver::{file_input, str_input};
use rustc::driver::session;
use syntax::diagnostic;
use syntax::ast;
use syntax::codemap;
use syntax::parse;

pub fn from_file(file: &Path) -> @ast::crate {
    parse::parse_crate_from_file(
        file, ~[], parse::new_parse_sess(None))
}

pub fn from_str(source: ~str) -> @ast::crate {
    parse::parse_crate_from_source_str(
        ~"-", @source, ~[], parse::new_parse_sess(None))
}

pub fn from_file_sess(sess: session::Session, file: &Path) -> @ast::crate {
    parse::parse_crate_from_file(
        file, cfg(sess, file_input(*file)), sess.parse_sess)
}

pub fn from_str_sess(sess: session::Session, source: ~str) -> @ast::crate {
    parse::parse_crate_from_source_str(
        ~"-", @source, cfg(sess, str_input(source)), sess.parse_sess)
}

fn cfg(sess: session::Session, input: driver::input) -> ast::crate_cfg {
    driver::default_configuration(sess, ~"rustdoc", input)
}
