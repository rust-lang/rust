#![feature(custom_attribute, test)]
#![feature(rustc_private)]
#![allow(unused_attributes)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate test;

use self::miri::interpreter;
use self::rustc::session::Session;
use self::rustc_driver::{driver, CompilerCalls};
use std::cell::RefCell;
use std::rc::Rc;
use std::env::var;
use test::Bencher;

pub struct MiriCompilerCalls<'a>(Rc<RefCell<&'a mut Bencher>>);

pub fn run(filename: &str, bencher: &mut Bencher) {
    let path = var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");
    rustc_driver::run_compiler(&[
        "miri".to_string(), format!("benches/{}.rs", filename), "--sysroot".to_string(), path.to_string(),
    ], &mut MiriCompilerCalls(Rc::new(RefCell::new(bencher))));
}

impl<'a> CompilerCalls<'a> for MiriCompilerCalls<'a> {
    fn build_controller(
        &mut self,
        _: &Session,
        _: &getopts::Matches
    ) -> driver::CompileController<'a> {
        let mut control: driver::CompileController<'a> = driver::CompileController::basic();

        let bencher = self.0.clone();

        control.after_analysis.callback = Box::new(move |state| {
            state.session.abort_if_errors();
            bencher.borrow_mut().iter(|| {
                interpreter::interpret_start_points(state.tcx.unwrap(), state.mir_map.unwrap());
            })
        });

        control
    }
}
