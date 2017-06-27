#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_metadata;
extern crate semver;
extern crate semverver;
extern crate syntax;

use semverver::semcheck::run_analysis;

use rustc::hir::def_id::*;
use rustc::session::{config, Session};
use rustc::session::config::{Input, ErrorOutputType};

use rustc_driver::{driver, CompilerCalls, RustcDefaultCalls, Compilation};

use std::path::PathBuf;
use std::process::Command;

use syntax::ast;

/// After the typechecker has finished it's work, we perform our checks.
///
/// To compare the two well-typed crates, we first find the aptly named crates `new` and `old`,
/// find their root modules and then proceed to walk their module trees.
fn callback(state: &driver::CompileState, version: &str) {
    let tcx = state.tcx.unwrap();
    let cstore = &tcx.sess.cstore;

    let (old_did, new_did) = if std::env::var("RUST_SEMVERVER_TEST").is_err() {
        // this is an actual program run
        let cnums = cstore
            .crates()
            .iter()
            .fold((None, None), |(o, n), crate_num| {
                let name = cstore.crate_name(*crate_num);
                if name == "old" {
                    (Some(*crate_num), n)
                } else if name == "new" {
                    (o, Some(*crate_num))
                } else {
                    (o, n)
                }
            });
        if let (Some(c0), Some(c1)) = cnums {
            (DefId {
                 krate: c0,
                 index: CRATE_DEF_INDEX,
             },
             DefId {
                 krate: c1,
                 index: CRATE_DEF_INDEX,
             })
        } else {
            tcx.sess.err("could not find crate `old` and/or `new`");
            return;
        }
    } else {
        // we are testing, so just fetch the *modules* `old` and `new` from a crate `oldandnew`
        let cnum = cstore
            .crates()
            .iter()
            .fold(None,
                  |k, crate_num| if cstore.crate_name(*crate_num) == "oldandnew" {
                      Some(*crate_num)
                  } else {
                      k
                  });

        let mod_did = if let Some(c) = cnum {
            DefId {
                krate: c,
                index: CRATE_DEF_INDEX,
            }
        } else {
            tcx.sess.err("could not find crate `oldandnew`");
            return;
        };

        let mut children = cstore.item_children(mod_did, tcx.sess);

        let dids = children
            .drain(..)
            .fold((None, None), |(o, n), child| {
                let child_name = String::from(&*child.ident.name.as_str());
                if child_name == "old" {
                    (Some(child.def.def_id()), n)
                } else if child_name == "new" {
                    (o, Some(child.def.def_id()))
                } else {
                    (o, n)
                }
            });

        if let (Some(o), Some(n)) = dids {
            (o, n)
        } else {
            tcx.sess.err("could not find module `new` and/or `old` in crate `oldandnew`");
            return;
        }
    };

    let changes = run_analysis(tcx, old_did, new_did);

    changes.output(tcx.sess, version);
}

/// Our wrapper to control compilation.
struct SemVerVerCompilerCalls {
    default: RustcDefaultCalls,
    version: String,
}

impl SemVerVerCompilerCalls {
    pub fn new(version: String) -> SemVerVerCompilerCalls {
        SemVerVerCompilerCalls {
            default: RustcDefaultCalls,
            version: version,
        }
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

        let old_callback = std::mem::replace(&mut controller.after_analysis.callback, box |_| {});
        let version = self.version.clone();
        controller.after_analysis.callback = box move |state| {
                                                     callback(state, &version);
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

        let version = if let Ok(ver) = std::env::var("RUST_SEMVER_CRATE_VERSION") {
            ver
        } else {
            std::process::exit(1);
        };

        let mut cc = SemVerVerCompilerCalls::new(version);
        let (result, _) = rustc_driver::run_compiler(&args, &mut cc, None, None);
        if let Err(count) = result {
            if count > 0 {
                std::process::exit(1);
            }
        }
    })
            .expect("rustc thread failed");
}
