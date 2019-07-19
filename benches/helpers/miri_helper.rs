extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate test;

use self::miri::eval_main;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc_interface::interface;
use rustc_driver::Compilation;
use crate::test::Bencher;

struct MiriCompilerCalls<'a> {
    bencher: &'a mut Bencher,
}

impl rustc_driver::Callbacks for MiriCompilerCalls<'_> {
    fn after_analysis(&mut self, compiler: &interface::Compiler) -> Compilation {
        compiler.session().abort_if_errors();

        compiler.global_ctxt().unwrap().peek_mut().enter(|tcx| {
            let (entry_def_id, _) = tcx.entry_fn(LOCAL_CRATE).expect(
                "no main or start function found",
            );

            self.bencher.iter(|| {
                let config = miri::MiriConfig { validate: true, args: vec![], seed: None };
                eval_main(tcx, entry_def_id, config);
            });
        });

        compiler.session().abort_if_errors();

        Compilation::Stop
    }
}

fn find_sysroot() -> String {
    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => {
            option_env!("RUST_SYSROOT")
                .expect(
                    "need to specify RUST_SYSROOT env var or use rustup or multirust",
                )
                .to_owned()
        }
    }
}

pub fn run(filename: &str, bencher: &mut Bencher) {
    let args = &[
        "miri".to_string(),
        format!("benches/helpers/{}.rs", filename),
        "--sysroot".to_string(),
        find_sysroot(),
    ];
    rustc_driver::run_compiler(args, &mut MiriCompilerCalls { bencher }, None, None).unwrap()
}
