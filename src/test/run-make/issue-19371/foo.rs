// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rustc;
extern crate rustc_trans;
extern crate syntax;

use rustc::session::{build_session, Session};
use rustc::session::config::{basic_options, build_configuration, OutputTypeExe};
use rustc_trans::driver::driver::{Input, StrInput, compile_input};
use syntax::diagnostics::registry::Registry;

fn main() {
    let src = r#"
    fn main() {}
    "#;

    let args = std::os::args();

    if args.len() < 4 {
        panic!("expected rustc path");
    }

    let tmpdir = Path::new(args[1].as_slice());

    let mut sysroot = Path::new(args[3].as_slice());
    sysroot.pop();
    sysroot.pop();

    compile(src.to_string(), tmpdir.join("out"), sysroot.clone());

    compile(src.to_string(), tmpdir.join("out"), sysroot.clone());
}

fn basic_sess(sysroot: Path) -> Session {
    let mut opts = basic_options();
    opts.output_types = vec![OutputTypeExe];
    opts.maybe_sysroot = Some(sysroot);

    let descriptions = Registry::new(&rustc::DIAGNOSTICS);
    let sess = build_session(opts, None, descriptions);
    sess
}

fn compile(code: String, output: Path, sysroot: Path) {
    let sess = basic_sess(sysroot);
    let cfg = build_configuration(&sess);

    compile_input(sess,
            cfg,
            &StrInput(code),
            &None,
            &Some(output),
            None);
}
