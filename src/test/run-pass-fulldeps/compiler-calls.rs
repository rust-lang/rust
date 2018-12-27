// Test that the CompilerCalls interface to the compiler works.

// ignore-cross-compile
// ignore-stage1

#![feature(rustc_private)]

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_codegen_utils;
extern crate syntax;
extern crate rustc_errors as errors;
extern crate rustc_metadata;

use rustc::session::Session;
use rustc::session::config::{self, Input};
use rustc_driver::{driver, CompilerCalls, Compilation};
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_metadata::cstore::CStore;
use syntax::ast;

use std::path::PathBuf;

struct TestCalls<'a> {
    count: &'a mut u32
}

impl<'a> CompilerCalls<'a> for TestCalls<'a> {
    fn early_callback(&mut self,
                      _: &getopts::Matches,
                      _: &config::Options,
                      _: &ast::CrateConfig,
                      _: &errors::registry::Registry,
                      _: config::ErrorOutputType)
                      -> Compilation {
        *self.count *= 2;
        Compilation::Continue
    }

    fn late_callback(&mut self,
                     _: &CodegenBackend,
                     _: &getopts::Matches,
                     _: &Session,
                     _: &CStore,
                     _: &Input,
                     _: &Option<PathBuf>,
                     _: &Option<PathBuf>)
                     -> Compilation {
        *self.count *= 3;
        Compilation::Stop
    }

    fn some_input(&mut self, input: Input, input_path: Option<PathBuf>)
                  -> (Input, Option<PathBuf>) {
        *self.count *= 5;
        (input, input_path)
    }

    fn no_input(&mut self,
                _: &getopts::Matches,
                _: &config::Options,
                _: &ast::CrateConfig,
                _: &Option<PathBuf>,
                _: &Option<PathBuf>,
                _: &errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        panic!("This shouldn't happen");
    }

    fn build_controller(self: Box<Self>,
                        _: &Session,
                        _: &getopts::Matches)
                        -> driver::CompileController<'a> {
        panic!("This shouldn't be called");
    }
}


fn main() {
    let mut count = 1;
    {
        let tc = TestCalls { count: &mut count };
        // we should never get use this filename, but lets make sure they are valid args.
        let args = vec!["compiler-calls".to_string(), "foo.rs".to_string()];
        syntax::with_globals(|| {
            rustc_driver::run_compiler(&args, Box::new(tc), None, None);
        });
    }
    assert_eq!(count, 30);
}
