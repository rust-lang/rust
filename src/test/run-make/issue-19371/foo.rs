// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_driver;
extern crate rustc_lint;
extern crate rustc_metadata;
extern crate syntax;

use rustc::session::{build_session, Session};
use rustc::session::config::{basic_options, build_configuration, Input, OutputType};
use rustc_driver::driver::{compile_input, CompileController};
use rustc_metadata::cstore::CStore;
use syntax::diagnostics::registry::Registry;
use syntax::parse::token;

use std::path::PathBuf;
use std::rc::Rc;

fn main() {
    let src = r#"
    fn main() {}
    "#;

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        panic!("expected rustc path");
    }

    let tmpdir = PathBuf::from(&args[1]);

    let mut sysroot = PathBuf::from(&args[3]);
    sysroot.pop();
    sysroot.pop();

    compile(src.to_string(), tmpdir.join("out"), sysroot.clone());

    compile(src.to_string(), tmpdir.join("out"), sysroot.clone());
}

fn basic_sess(sysroot: PathBuf) -> (Session, Rc<CStore>) {
    let mut opts = basic_options();
    opts.output_types.insert(OutputType::Exe, None);
    opts.maybe_sysroot = Some(sysroot);

    let descriptions = Registry::new(&rustc::DIAGNOSTICS);
    let cstore = Rc::new(CStore::new(token::get_ident_interner()));
    let sess = build_session(opts, None, descriptions, cstore.clone());
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));
    (sess, cstore)
}

fn compile(code: String, output: PathBuf, sysroot: PathBuf) {
    let (sess, cstore) = basic_sess(sysroot);
    let cfg = build_configuration(&sess);
    let control = CompileController::basic();

    compile_input(sess, &cstore,
            cfg,
            &Input::Str(code),
            &None,
            &Some(output),
            None,
            control);
}
