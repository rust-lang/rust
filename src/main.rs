// error-pattern:yummy
#![feature(box_syntax)]
#![feature(rustc_private)]
#![feature(slice_patterns)]

extern crate rustc_driver;
extern crate getopts;
extern crate rustc;
extern crate syntax;
extern crate rustc_plugin;
extern crate clippy_lints;
extern crate rustc_serialize;

use rustc_driver::{driver, CompilerCalls, RustcDefaultCalls, Compilation};
use rustc::session::{config, Session};
use rustc::session::config::{Input, ErrorOutputType};
use syntax::diagnostics;
use std::path::PathBuf;
use std::process::Command;

use clippy_lints::utils::cargo;

struct ClippyCompilerCalls(RustcDefaultCalls);

impl std::default::Default for ClippyCompilerCalls {
    fn default() -> Self {
        Self::new()
    }
}

impl ClippyCompilerCalls {
    fn new() -> Self {
        ClippyCompilerCalls(RustcDefaultCalls)
    }
}

impl<'a> CompilerCalls<'a> for ClippyCompilerCalls {
    fn early_callback(&mut self,
                      matches: &getopts::Matches,
                      sopts: &config::Options,
                      descriptions: &diagnostics::registry::Registry,
                      output: ErrorOutputType)
                      -> Compilation {
        self.0.early_callback(matches, sopts, descriptions, output)
    }
    fn no_input(&mut self,
                matches: &getopts::Matches,
                sopts: &config::Options,
                odir: &Option<PathBuf>,
                ofile: &Option<PathBuf>,
                descriptions: &diagnostics::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        self.0.no_input(matches, sopts, odir, ofile, descriptions)
    }
    fn late_callback(&mut self,
                     matches: &getopts::Matches,
                     sess: &Session,
                     input: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        self.0.late_callback(matches, sess, input, odir, ofile)
    }
    fn build_controller(&mut self, sess: &Session, matches: &getopts::Matches) -> driver::CompileController<'a> {
        let mut control = self.0.build_controller(sess, matches);

        let old = std::mem::replace(&mut control.after_parse.callback, box |_| {});
        control.after_parse.callback = Box::new(move |state| {
            {
                let mut registry = rustc_plugin::registry::Registry::new(state.session, state.krate.as_ref().expect("at this compilation stage the krate must be parsed"));
                registry.args_hidden = Some(Vec::new());
                clippy_lints::register_plugins(&mut registry);

                let rustc_plugin::registry::Registry { early_lint_passes, late_lint_passes, lint_groups, llvm_passes, attributes, mir_passes, .. } = registry;
                let sess = &state.session;
                let mut ls = sess.lint_store.borrow_mut();
                for pass in early_lint_passes {
                    ls.register_early_pass(Some(sess), true, pass);
                }
                for pass in late_lint_passes {
                    ls.register_late_pass(Some(sess), true, pass);
                }

                for (name, to) in lint_groups {
                    ls.register_group(Some(sess), true, name, to);
                }

                sess.plugin_llvm_passes.borrow_mut().extend(llvm_passes);
                sess.mir_passes.borrow_mut().extend(mir_passes);
                sess.plugin_attributes.borrow_mut().extend(attributes);
            }
            old(state);
        });

        control
    }
}

use std::path::Path;

pub fn main() {
    use std::env;

    if env::var("CLIPPY_DOGFOOD").map(|_| true).unwrap_or(false) {
        panic!("yummy");
    }

    let dep_path = env::current_dir().expect("current dir is not readable").join("target").join("debug").join("deps");

    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    let sys_root = match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => option_env!("SYSROOT").map(|s| s.to_owned())
                                   .or(Command::new("rustc").arg("--print")
                                                            .arg("sysroot")
                                                            .output().ok()
                                                            .and_then(|out| String::from_utf8(out.stdout).ok())
                                                            .map(|s| s.trim().to_owned())
                                                            )
                .expect("need to specify SYSROOT env var during clippy compilation, or use rustup or multirust"),
    };

    if let Some("clippy") = std::env::args().nth(1).as_ref().map(AsRef::as_ref) {
        let mut metadata = cargo::metadata().expect("could not obtain cargo metadata");
        assert_eq!(metadata.version, 1);
        for target in metadata.packages.remove(0).targets {
            let args = std::env::args().skip(2);
            if let Some(first) = target.kind.get(0) {
                if target.kind.len() > 1 || first.ends_with("lib") {
                    if let Err(code) = process(std::iter::once("--lib".to_owned()).chain(args), &dep_path, &sys_root) {
                        std::process::exit(code);
                    }
                } else if first == "bin" {
                    if let Err(code) = process(vec!["--bin".to_owned(), target.name].into_iter().chain(args), &dep_path, &sys_root) {
                        std::process::exit(code);
                    }
                }
            } else {
                panic!("badly formatted cargo metadata: target::kind is an empty array");
            }
        }
    } else {
        let mut args: Vec<String> = if env::args().any(|s| s == "--sysroot") {
            env::args().collect()
        } else {
            env::args().chain(Some("--sysroot".to_owned())).chain(Some(sys_root)).collect()
        };

        args.extend_from_slice(&["--cfg".to_owned(), r#"feature="clippy""#.to_owned()]);

        let (result, _) = rustc_driver::run_compiler(&args, &mut ClippyCompilerCalls::new());

        if let Err(err_count) = result {
            if err_count > 0 {
                std::process::exit(1);
            }
        }
    }
}

fn process<P, I>(old_args: I, dep_path: P, sysroot: &str) -> Result<(), i32>
    where P: AsRef<Path>, I: Iterator<Item=String> {

    let mut args = vec!["rustc".to_owned()];

    let mut found_dashes = false;
    for arg in old_args {
        found_dashes |= arg == "--";
        args.push(arg);
    }
    if !found_dashes {
        args.push("--".to_owned());
    }
    args.push("-L".to_owned());
    args.push(dep_path.as_ref().to_string_lossy().into_owned());
    args.push(String::from("--sysroot"));
    args.push(sysroot.to_owned());
    args.push("-Zno-trans".to_owned());
    args.push("--cfg".to_owned());
    args.push(r#"feature="clippy""#.to_owned());

    let path = std::env::current_exe().expect("current executable path invalid");
    let exit_status = std::process::Command::new("cargo")
        .args(&args)
        .env("RUSTC", path)
        .spawn().expect("could not run cargo")
        .wait().expect("failed to wait for cargo?");

    if exit_status.success() {
        Ok(())
    } else {
        Err(exit_status.code().unwrap_or(-1))
    }
}
