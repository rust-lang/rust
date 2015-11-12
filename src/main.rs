#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_driver;
extern crate rustc_mir;
extern crate syntax;

mod interpreter;

use rustc::session::Session;
use rustc_driver::{driver, CompilerCalls, Compilation};

struct MiriCompilerCalls;

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn build_controller(&mut self, _: &Session) -> driver::CompileController<'a> {
        let mut control = driver::CompileController::basic();
        control.after_analysis.stop = Compilation::Stop;

        control.after_analysis.callback = Box::new(|state| {
            interpreter::interpret_start_points(state.tcx.unwrap(), state.mir_map.unwrap());
        });

        control
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    rustc_driver::run_compiler(&args, &mut MiriCompilerCalls);
}
