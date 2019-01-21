#![feature(rustc_private)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_metadata;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_codegen_utils;
extern crate env_logger;
extern crate log_settings;
extern crate syntax;

#[macro_use]
extern crate log;

use std::path::PathBuf;
use std::str::FromStr;
use std::env;

use rustc::session::Session;
use rustc_metadata::cstore::CStore;
use rustc_driver::{Compilation, CompilerCalls, RustcDefaultCalls};
use rustc_driver::driver::{CompileState, CompileController};
use rustc::session::config::{self, Input, ErrorOutputType};
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc::hir::def_id::LOCAL_CRATE;
use syntax::ast;

struct MiriCompilerCalls {
    default: Box<RustcDefaultCalls>,

    /// Whether to enforce the validity invariant.
    validate: bool,
}

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn early_callback(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        descriptions: &rustc_errors::registry::Registry,
        output: ErrorOutputType,
    ) -> Compilation {
        self.default.early_callback(
            matches,
            sopts,
            cfg,
            descriptions,
            output,
        )
    }
    fn no_input(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
        descriptions: &rustc_errors::registry::Registry,
    ) -> Option<(Input, Option<PathBuf>)> {
        self.default.no_input(
            matches,
            sopts,
            cfg,
            odir,
            ofile,
            descriptions,
        )
    }
    fn late_callback(
        &mut self,
        codegen_backend: &CodegenBackend,
        matches: &getopts::Matches,
        sess: &Session,
        cstore: &CStore,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
    ) -> Compilation {
        self.default.late_callback(codegen_backend, matches, sess, cstore, input, odir, ofile)
    }
    fn build_controller(
        self: Box<Self>,
        sess: &Session,
        matches: &getopts::Matches,
    ) -> CompileController<'a> {
        let this = *self;
        let mut control = this.default.build_controller(sess, matches);
        control.after_hir_lowering.callback = Box::new(after_hir_lowering);
        let validate = this.validate;
        control.after_analysis.callback =
            Box::new(move |state| after_analysis(state, validate));
        control.after_analysis.stop = Compilation::Stop;
        control
    }
}

fn after_hir_lowering(state: &mut CompileState) {
    let attr = (
        String::from("miri"),
        syntax::feature_gate::AttributeType::Whitelisted,
    );
    state.session.plugin_attributes.borrow_mut().push(attr);
}

fn after_analysis<'a, 'tcx>(
    state: &mut CompileState<'a, 'tcx>,
    validate: bool,
) {
    init_late_loggers();
    state.session.abort_if_errors();

    let tcx = state.tcx.unwrap();


    let (entry_def_id, _) = tcx.entry_fn(LOCAL_CRATE).expect("no main function found!");

    miri::eval_main(tcx, entry_def_id, validate);

    state.session.abort_if_errors();
}

fn init_early_loggers() {
    // Notice that our `extern crate log` is NOT the same as rustc's!  So we have to initialize
    // them both.  We always initialize miri early.
    let env = env_logger::Env::new().filter("MIRI_LOG").write_style("MIRI_LOG_STYLE");
    env_logger::init_from_env(env);
    // We only initialize rustc if the env var is set (so the user asked for it).
    // If it is not set, we avoid initializing now so that we can initialize
    // later with our custom settings, and NOT log anything for what happens before
    // miri gets started.
    if env::var("RUST_LOG").is_ok() {
        rustc_driver::init_rustc_env_logger();
    }
}

fn init_late_loggers() {
    // Initializing loggers right before we start evaluation.  We overwrite the RUST_LOG
    // env var if it is not set, control it based on MIRI_LOG.
    if let Ok(var) = env::var("MIRI_LOG") {
        if env::var("RUST_LOG").is_err() {
            // We try to be a bit clever here: If MIRI_LOG is just a single level
            // used for everything, we only apply it to the parts of rustc that are
            // CTFE-related.  Otherwise, we use it verbatim for RUST_LOG.
            // This way, if you set `MIRI_LOG=trace`, you get only the right parts of
            // rustc traced, but you can also do `MIRI_LOG=miri=trace,rustc_mir::interpret=debug`.
            if log::Level::from_str(&var).is_ok() {
                env::set_var("RUST_LOG",
                    &format!("rustc::mir::interpret={0},rustc_mir::interpret={0}", var));
            } else {
                env::set_var("RUST_LOG", &var);
            }
            rustc_driver::init_rustc_env_logger();
        }
    }

    // If MIRI_BACKTRACE is set and RUST_CTFE_BACKTRACE is not, set RUST_CTFE_BACKTRACE.
    // Do this late, so we really only apply this to miri's errors.
    if let Ok(var) = env::var("MIRI_BACKTRACE") {
        if env::var("RUST_CTFE_BACKTRACE") == Err(env::VarError::NotPresent) {
            env::set_var("RUST_CTFE_BACKTRACE", &var);
        }
    }
}

fn find_sysroot() -> String {
    if let Ok(sysroot) = std::env::var("MIRI_SYSROOT") {
        return sysroot;
    }

    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => {
            option_env!("RUST_SYSROOT")
                .expect(
                    "Could not find sysroot. Either set MIRI_SYSROOT at run-time, or at \
                     build-time specify RUST_SYSROOT env var or use rustup or multirust",
                )
                .to_owned()
        }
    }
}

fn main() {
    init_early_loggers();
    let mut args: Vec<String> = std::env::args().collect();

    // Parse our own -Z flags and remove them before rustc gets their hand on them.
    let mut validate = true;
    args.retain(|arg| {
        match arg.as_str() {
            "-Zmiri-disable-validation" => {
                validate = false;
                false
            },
            _ => true
        }
    });

    // Determine sysroot and let rustc know about it
    let sysroot_flag = String::from("--sysroot");
    if !args.contains(&sysroot_flag) {
        args.push(sysroot_flag);
        args.push(find_sysroot());
    }
    // Finally, add the default flags all the way in the beginning, but after the binary name.
    args.splice(1..1, miri::miri_default_args().iter().map(ToString::to_string));

    trace!("rustc arguments: {:?}", args);
    let result = rustc_driver::run(move || {
        rustc_driver::run_compiler(&args, Box::new(MiriCompilerCalls {
            default: Box::new(RustcDefaultCalls),
            validate,
        }), None, None)
    });
    std::process::exit(result as i32);
}
