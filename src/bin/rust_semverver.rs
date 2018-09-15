#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate env_logger;
extern crate getopts;
#[macro_use]
extern crate log;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_codegen_utils;
extern crate rustc_metadata;
extern crate semverver;
extern crate syntax;

use semverver::semcheck::run_analysis;

use rustc::hir::def_id::*;
use rustc_metadata::cstore::CStore;
use rustc::session::{config, Session};
use rustc::session::config::{Input, ErrorOutputType};
use rustc::middle::cstore::ExternCrate;

use rustc_driver::{driver, CompilerCalls, RustcDefaultCalls, Compilation};

use rustc_codegen_utils::codegen_backend::CodegenBackend;

use std::path::PathBuf;
use std::process::Command;

use syntax::ast;
use syntax::source_map::Pos;

/// After the typechecker has finished it's work, perform our checks.
fn callback(state: &driver::CompileState, version: &str, verbose: bool) {
    let tcx = state.tcx.unwrap();

    // To select the old and new crates we look at the position of the declaration in the
    // source file.  The first one will be the `old` and the other will be `new`.  This is
    // unfortunately a bit hacky... See issue #64 for details.

    let mut crates: Vec<_> = tcx
        .crates()
        .iter()
        .flat_map(|crate_num| {
            let def_id = DefId {
                krate: *crate_num,
                index: CRATE_DEF_INDEX,
            };

            match *tcx.extern_crate(def_id) {
                Some(ExternCrate { span, direct: true, ..}) if span.data().lo.to_usize() > 0 =>
                    Some((span.data().lo.to_usize(), def_id)),
                _ => None,
            }
        })
        .collect();

    crates.sort_by_key(|&(span_lo, _)| span_lo);

    match crates.as_slice() {
        &[(_, old_def_id), (_, new_def_id)] => {
            debug!("running semver analysis");
            let changes = run_analysis(tcx, old_def_id, new_def_id);

            changes.output(tcx.sess, version, verbose);
        }
        _ => {
            tcx.sess.err("could not find crate old and new crates");
        }
    }
}

/// A wrapper to control compilation.
struct SemVerVerCompilerCalls {
    /// The wrapped compilation handle.
    default: Box<RustcDefaultCalls>,
    /// The version of the old crate.
    version: String,
    /// The output mode.
    verbose: bool,
}

impl SemVerVerCompilerCalls {
    /// Construct a new compilation wrapper, given a version string.
    pub fn new(version: String, verbose: bool) -> Box<SemVerVerCompilerCalls> {
        Box::new(SemVerVerCompilerCalls {
            default: Box::new(RustcDefaultCalls),
            version,
            verbose,
        })
    }

    pub fn get_default(&self) -> Box<RustcDefaultCalls> {
        self.default.clone()
    }

    pub fn get_version(&self) -> &String {
        &self.version
    }
}

impl<'a> CompilerCalls<'a> for SemVerVerCompilerCalls {
    fn early_callback(&mut self,
                      matches: &getopts::Matches,
                      sopts: &config::Options,
                      cfg: &ast::CrateConfig,
                      descriptions: &rustc_errors::registry::Registry,
                      output: ErrorOutputType)
                      -> Compilation {
        debug!("running rust-semverver early_callback");
        self.default
            .early_callback(matches, sopts, cfg, descriptions, output)
    }

    fn no_input(&mut self,
                matches: &getopts::Matches,
                sopts: &config::Options,
                cfg: &ast::CrateConfig,
                odir: &Option<PathBuf>,
                ofile: &Option<PathBuf>,
                descriptions: &rustc_errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        debug!("running rust-semverver no_input");
        self.default
            .no_input(matches, sopts, cfg, odir, ofile, descriptions)
    }

    fn late_callback(&mut self,
                     trans_crate: &CodegenBackend,
                     matches: &getopts::Matches,
                     sess: &Session,
                     cstore: &CStore,
                     input: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        debug!("running rust-semverver late_callback");
        self.default
            .late_callback(trans_crate, matches, sess, cstore, input, odir, ofile)
    }

    fn build_controller(self: Box<SemVerVerCompilerCalls>,
                        sess: &Session,
                        matches: &getopts::Matches)
                        -> driver::CompileController<'a> {
        let default = self.get_default();
        let version = self.get_version().clone();
        let SemVerVerCompilerCalls { verbose, .. } = *self;
        let mut controller = CompilerCalls::build_controller(default, sess, matches);
        let old_callback =
            std::mem::replace(&mut controller.after_analysis.callback, box |_| {});

        controller.after_analysis.callback = box move |state| {
            debug!("running rust-semverver after_analysis callback");
            callback(state, &version, verbose);
            debug!("running other after_analysis callback");
            old_callback(state);
        };
        controller.after_analysis.stop = Compilation::Stop;

        controller
    }
}

/// Main routine.
///
/// Find the sysroot before passing our args to the compiler driver, after registering our custom
/// compiler driver.
fn main() {
    if env_logger::try_init().is_err() {
        eprintln!("ERROR: could not initialize logger");
    }

    debug!("running rust-semverver compiler driver");

    let home = option_env!("RUSTUP_HOME");
    let toolchain = option_env!("RUSTUP_TOOLCHAIN");
    let sys_root = if let (Some(home), Some(toolchain)) = (home, toolchain) {
        format!("{}/toolchains/{}", home, toolchain)
    } else {
        option_env!("SYSROOT")
            .map(|s| s.to_owned())
            .or_else(|| {
                Command::new("rustc")
                    .args(&["--print", "sysroot"])
                    .output()
                    .ok()
                    .and_then(|out| String::from_utf8(out.stdout).ok())
                    .map(|s| s.trim().to_owned())
            })
            .expect("need to specify SYSROOT env var during compilation, or use rustup")
    };

    let result = rustc_driver::run(|| {
        let args: Vec<String> = if std::env::args().any(|s| s == "--sysroot") {
            std::env::args().collect()
        } else {
            std::env::args()
                .chain(Some("--sysroot".to_owned()))
                .chain(Some(sys_root))
                .collect()
        };

        let version = if let Ok(ver) = std::env::var("RUST_SEMVER_CRATE_VERSION") {
            ver
        } else {
            "no_version".to_owned()
        };

        let verbose = std::env::var("RUST_SEMVER_VERBOSE") == Ok("true".to_string());

        let cc = SemVerVerCompilerCalls::new(version, verbose);
        rustc_driver::run_compiler(&args, cc, None, None)
    });

    std::process::exit(result as i32);
}
