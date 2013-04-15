// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::driver::{driver, session};
use syntax::{ast, diagnostic};
use syntax::parse::token;
use core::path::Path;
use core::option::*;

/// True if the file at path `p` contains a `main` function
pub fn has_main_fn(p: &Path, binary: ~str) -> bool {
    let input = driver::file_input(copy *p);
    let options = @session::options {
        binary: copy binary,
        crate_type: session::bin_crate,
        .. *session::basic_options()
    };
    // Should probably take a session as an argument
    let sess = driver::build_session(options, diagnostic::emit);
    let cfg = driver::build_configuration(sess, binary, input);
    let (crate, _) = driver::compile_upto(sess, cfg, input, driver::cu_parse, None);
    let mut has_main = false;
    for crate.node.module.items.each() |it| {
        match it.node {
            ast::item_fn(*) if it.ident == token::special_idents::main =>
                has_main = true,
            _ => ()
        }
    }
    has_main
}
