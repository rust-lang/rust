//! The only difference between this and cg_clif.rs is that this binary defaults to using cg_llvm
//! instead of cg_clif and requires `--clif` to use cg_clif and that this binary doesn't have JIT
//! support.
//! This is necessary as with Cargo `RUSTC` applies to both target crates and host crates. The host
//! crates must be built with cg_llvm as we are currently building a sysroot for cg_clif.
//! `RUSTFLAGS` however is only applied to target crates, so `--clif` would only be passed to the
//! target crates.

#![feature(rustc_private)]

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_target;

use std::path::PathBuf;

use rustc_interface::interface;
use rustc_session::config::ErrorOutputType;
use rustc_session::early_error;
use rustc_target::spec::PanicStrategy;

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

pub struct CraneliftPassesCallbacks {
    use_clif: bool,
}

impl rustc_driver::Callbacks for CraneliftPassesCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        if !self.use_clif {
            config.opts.maybe_sysroot = Some(PathBuf::from(find_sysroot()));
            return;
        }

        config.opts.cg.panic = Some(PanicStrategy::Abort);
        config.opts.debugging_opts.panic_abort_tests = true;
        config.opts.maybe_sysroot = Some(
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("build_sysroot")
                .join("sysroot"),
        );
    }
}

fn main() {
    rustc_driver::init_rustc_env_logger();
    rustc_driver::install_ice_hook();
    let exit_code = rustc_driver::catch_with_exit_code(|| {
        let mut use_clif = false;

        let args = std::env::args_os()
            .enumerate()
            .map(|(i, arg)| {
                arg.into_string().unwrap_or_else(|arg| {
                    early_error(
                        ErrorOutputType::default(),
                        &format!("Argument {} is not valid Unicode: {:?}", i, arg),
                    )
                })
            })
            .filter(|arg| {
                if arg == "--clif" {
                    use_clif = true;
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        let mut callbacks = CraneliftPassesCallbacks { use_clif };

        let mut run_compiler = rustc_driver::RunCompiler::new(&args, &mut callbacks);
        if use_clif {
            run_compiler.set_make_codegen_backend(Some(Box::new(move |_| {
                Box::new(rustc_codegen_cranelift::CraneliftCodegenBackend {
                    config: rustc_codegen_cranelift::BackendConfig { use_jit: false },
                })
            })));
        }
        run_compiler.run()
    });
    std::process::exit(exit_code)
}
