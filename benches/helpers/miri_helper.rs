extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate test;

use self::miri::eval_main;
use self::rustc::session::Session;
use self::rustc_driver::{driver, CompilerCalls, Compilation};
use std::cell::RefCell;
use std::rc::Rc;
use test::Bencher;

pub struct MiriCompilerCalls<'a>(Rc<RefCell<&'a mut Bencher>>);

fn find_sysroot() -> String {
    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => option_env!("RUST_SYSROOT")
            .expect("need to specify RUST_SYSROOT env var or use rustup or multirust")
            .to_owned(),
    }
}

pub fn run(filename: &str, bencher: &mut Bencher) {
    let args = &[
        "miri".to_string(),
        format!("benches/helpers/{}.rs", filename),
        "--sysroot".to_string(),
        find_sysroot()
    ];
    let compiler_calls = &mut MiriCompilerCalls(Rc::new(RefCell::new(bencher)));
    rustc_driver::run_compiler(args, compiler_calls);
}

impl<'a> CompilerCalls<'a> for MiriCompilerCalls<'a> {
    fn build_controller(
        &mut self,
        _: &Session,
        _: &getopts::Matches
    ) -> driver::CompileController<'a> {
        let mut control: driver::CompileController<'a> = driver::CompileController::basic();

        let bencher = self.0.clone();

        control.after_analysis.stop = Compilation::Stop;
        control.after_analysis.callback = Box::new(move |state| {
            state.session.abort_if_errors();

            let tcx = state.tcx.unwrap();
            let mir_map = state.mir_map.unwrap();
            let (node_id, _) = state.session.entry_fn.borrow()
                .expect("no main or start function found");

            bencher.borrow_mut().iter(|| { eval_main(tcx, mir_map, node_id); });

            state.session.abort_if_errors();
        });

        control
    }
}
