#![feature(custom_attribute, test)]
#![feature(rustc_private)]
#![allow(unused_attributes)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;

use miri::interpreter;
use rustc::session::Session;
use rustc_driver::{driver, CompilerCalls};
use std::cell::RefCell;
use std::rc::Rc;

extern crate test;
use test::Bencher;

mod smoke_helper;

#[bench]
fn noop(bencher: &mut Bencher) {
    bencher.iter(|| {
        smoke_helper::main();
    })
}

/*
// really slow
#[bench]
fn noop_miri_full(bencher: &mut Bencher) {
    let path = std::env::var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");
    bencher.iter(|| {
        let mut process = std::process::Command::new("target/release/miri");
        process.arg("benches/smoke_helper.rs")
               .arg("--sysroot").arg(&path);
        let output = process.output().unwrap();
        if !output.status.success() {
            println!("{}", String::from_utf8(output.stdout).unwrap());
            println!("{}", String::from_utf8(output.stderr).unwrap());
            panic!("failed to run miri");
        }
    })
}
*/

#[bench]
fn noop_miri_interpreter(bencher: &mut Bencher) {
    let path = std::env::var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");
    rustc_driver::run_compiler(&[
        "miri".to_string(), "benches/smoke_helper.rs".to_string(), "--sysroot".to_string(), path.to_string(),
    ], &mut MiriCompilerCalls(Rc::new(RefCell::new(bencher))));
}

struct MiriCompilerCalls<'a>(Rc<RefCell<&'a mut Bencher>>);

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
