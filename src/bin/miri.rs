#![feature(rustc_private, custom_attribute)]
#![allow(unused_attributes)]

extern crate miri;
extern crate rustc;
extern crate rustc_driver;

use miri::interpreter;
use rustc::session::Session;
use rustc_driver::{driver, CompilerCalls};

struct MiriCompilerCalls;

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn build_controller(&mut self, _: &Session) -> driver::CompileController<'a> {
        let mut control = driver::CompileController::basic();

        control.after_analysis.callback = Box::new(|state| {
            state.session.abort_if_errors();
            interpreter::interpret_start_points(state.tcx.unwrap(), state.mir_map.unwrap());
        });

        control
    }
}

#[miri_run]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    rustc_driver::run_compiler(&args, &mut MiriCompilerCalls);
}
