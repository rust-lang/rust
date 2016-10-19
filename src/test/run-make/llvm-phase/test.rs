// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(plugin, rustc_private, box_syntax)]

extern crate rustc;
extern crate rustc_driver;
extern crate rustc_llvm;
extern crate rustc_trans;
#[macro_use] extern crate syntax;
extern crate getopts;

use rustc_driver::{CompilerCalls, Compilation};
use rustc_driver::driver::CompileController;
use rustc_trans::ModuleSource;
use rustc::session::Session;
use syntax::codemap::FileLoader;
use std::env;
use std::io;
use std::path::{PathBuf, Path};

struct JitLoader;

impl FileLoader for JitLoader {
    fn file_exists(&self, _: &Path) -> bool { true }
    fn abs_path(&self, _: &Path) -> Option<PathBuf> { None }
    fn read_file(&self, _: &Path) -> io::Result<String> {
        Ok(r#"
#[no_mangle]
pub fn test_add(a: i32, b: i32) -> i32 { a + b }
"#.to_string())
    }
}

#[derive(Copy, Clone)]
struct JitCalls;

impl<'a> CompilerCalls<'a> for JitCalls {
    fn build_controller(&mut self,
                        _: &Session,
                        _: &getopts::Matches)
                        -> CompileController<'a> {
        let mut cc = CompileController::basic();
        cc.after_llvm.stop = Compilation::Stop;
        cc.after_llvm.run_callback_on_error = true;
        cc.after_llvm.callback = Box::new(|state| {
            state.session.abort_if_errors();
            let trans = state.trans.unwrap();
            assert_eq!(trans.modules.len(), 1);
            let rs_llmod = match trans.modules[0].source {
                ModuleSource::Preexisting(_) => unimplemented!(),
                ModuleSource::Translated(llvm) => llvm.llmod,
            };
            unsafe { rustc_llvm::LLVMDumpModule(rs_llmod) };
        });
        cc
    }
}

fn main() {
    use rustc_driver;

    let mut path = match std::env::args().nth(2) {
        Some(path) => PathBuf::from(&path),
        None => panic!("missing rustc path")
    };

    // Remove two segments from rustc path to get sysroot.
    path.pop();
    path.pop();

    let mut args: Vec<String> =
        format!("_ _ --sysroot {} --crate-type dylib", path.to_str().unwrap())
        .split(' ').map(|s| s.to_string()).collect();
    args.push("--out-dir".to_string());
    args.push(env::var("TMPDIR").unwrap());

    let (result, _) = rustc_driver::run_compiler(
        &args, &mut JitCalls, Some(box JitLoader), None);
    if let Err(n) = result {
        panic!("Error {}", n);
    }
}
