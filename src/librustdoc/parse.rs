// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! AST-parsing helpers

use core::prelude::*;

use rustc::driver::driver::{file_input, str_input};
use rustc::driver::driver;
use rustc::driver::session;
use syntax::ast;
use syntax::parse;

pub fn from_file(file: &Path) -> @ast::crate {
    parse::parse_crate_from_file(
        file, ~[], parse::new_parse_sess(None))
}

pub fn from_str(source: @str) -> @ast::crate {
    parse::parse_crate_from_source_str(
        @"-", source, ~[], parse::new_parse_sess(None))
}

pub fn from_file_sess(sess: session::Session, file: &Path) -> @ast::crate {
    parse::parse_crate_from_file(
        file, cfg(sess, file_input(copy *file)), sess.parse_sess)
}

pub fn from_str_sess(sess: session::Session, source: @str) -> @ast::crate {
    parse::parse_crate_from_source_str(
        @"-", source, cfg(sess, str_input(source)), sess.parse_sess)
}

fn cfg(sess: session::Session, input: driver::input) -> ast::crate_cfg {
    driver::build_configuration(sess, @"rustdoc", &input)
}
