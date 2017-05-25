#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate cargo_metadata;
extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate syntax;

use rustc::session::{config, Session};
use rustc::session::config::{Input, ErrorOutputType};

use rustc_driver::{driver, CompilerCalls, RustcDefaultCalls, Compilation};

use std::io::{self, Write};
use std::path::PathBuf;

use syntax::ast;

struct SemVerVerCompilerCalls {
    default: RustcDefaultCalls,
}

impl SemVerVerCompilerCalls {
    pub fn new() -> SemVerVerCompilerCalls {
        SemVerVerCompilerCalls { default: RustcDefaultCalls }
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
        self.default
            .no_input(matches, sopts, cfg, odir, ofile, descriptions)
    }

    fn late_callback(&mut self,
                     matches: &getopts::Matches,
                     sess: &Session,
                     input: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        self.default
            .late_callback(matches, sess, input, odir, ofile)
    }

    fn build_controller(&mut self,
                        sess: &Session,
                        matches: &getopts::Matches)
                        -> driver::CompileController<'a> {
        let mut controller = self.default.build_controller(sess, matches);

        let old_callback = std::mem::replace(&mut controller.after_hir_lowering.callback,
                                             box |_| {});
        controller.after_hir_lowering.callback = box move |state| { old_callback(state); };

        controller
    }
}

const CARGO_SEMVER_HELP: &str = r#"Checks a package's SemVer compatibility with already published versions.

Usage:
    cargo semver [options] [--] [<opts>...]

Common options:
    -h, --help               Print this message
    --features               Features to compile for the package
    -V, --version            Print version info and exit

Other options are the same as `cargo rustc`.
"#;

fn help() {
    println!("{}", CARGO_SEMVER_HELP);
}

fn version() {
    println!("{}", env!("CARGO_PKG_VERSION"));
}

pub fn main() {
    // TODO: use getopt, as we import it anyway
    // TODO: maybe don't use cargo_metadata, as it pulls in tons of deps

    if std::env::args().any(|arg| arg == "-h" || arg == "--help") {
        help();
        return;
    }

    if std::env::args().any(|arg| arg == "-V" || arg == "--version") {
        version();
        return;
    }

    if std::env::args()
           .nth(1)
           .map(|arg| arg == "semver")
           .unwrap_or(false) {
        // first run (we blatantly copy clippy's code structure here)
        let manifest_path_arg = std::env::args()
            .skip(2)
            .find(|val| val.starts_with("--manifest-path="));

        let mut metadata = if let Ok(data) =
            cargo_metadata::metadata(manifest_path_arg.as_ref().map(AsRef::as_ref)) {
            data
        } else {
            let _ = io::stderr()
                .write_fmt(format_args!("error: could not obtain cargo metadata.\n"));
            std::process::exit(1);
        };
    } else {
        // second run TODO: elaborate etc
    }

    rustc_driver::in_rustc_thread(|| {
        let args: Vec<String> = std::env::args().collect(); // TODO: look at clippy here

        // let checks_enabled = std::env::args().any(|s| s == "-Zno-trans");

        let mut cc = SemVerVerCompilerCalls::new(); // TODO: use `checks_enabled`
        // TODO: the second result is a `Session` - maybe we'll need it
        let (result, _) = rustc_driver::run_compiler(&args, &mut cc, None, None);

        if let Err(count) = result {
            if count > 0 {
                std::process::exit(1);
            }
        }
    })
            .expect("rustc thread failed");
}
