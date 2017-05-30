#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_metadata;
extern crate syntax;

use rustc::hir::def_id::*;
use rustc::session::{config, Session};
use rustc::session::config::{Input, ErrorOutputType};

use rustc_driver::{driver, CompilerCalls, RustcDefaultCalls, Compilation};

use std::path::PathBuf;
use std::process::Command;

use syntax::ast;

fn callback(state: &driver::CompileState) {
    let tcx = state.tcx.unwrap();
    let cstore = &tcx.sess.cstore;

    let cnums = cstore.crates().iter().fold((None, None), |(n, o), crate_num| {
        let name = cstore.crate_name(*crate_num);
        if name == "new" {
            (Some(*crate_num), o)
        } else if name == "old" {
            (n, Some(*crate_num))
        } else {
            (n, o)
        }
    });

    let new_did = DefId { krate: cnums.0.unwrap(), index: CRATE_DEF_INDEX };
    let old_did = DefId { krate: cnums.1.unwrap(), index: CRATE_DEF_INDEX };

    println!("new: {:?}, old: {:?}", new_did, old_did);
}

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

        let old_callback = std::mem::replace(&mut controller.after_analysis.callback,
                                             box |_| {});
        controller.after_analysis.callback = box move |state| {
            callback(state);
            old_callback(state);
        };

        controller
    }
}

fn main() {
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

    rustc_driver::in_rustc_thread(|| {
        let args: Vec<String> = if std::env::args().any(|s| s == "--sysroot") {
            std::env::args().collect()
        } else {
            std::env::args()
                .chain(Some("--sysroot".to_owned()))
                .chain(Some(sys_root))
                .collect()
        };

        let mut cc = SemVerVerCompilerCalls::new();
        let (result, _) = rustc_driver::run_compiler(&args, &mut cc, None, None);
        if let Err(count) = result {
            if count > 0 {
                std::process::exit(1);
            }
        }
    })
    .expect("rustc thread failed");
}
