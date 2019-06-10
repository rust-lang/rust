#![feature(rustc_private)]
#![feature(result_map_or_else)]

extern crate rustc;
extern crate rustc_codegen_utils;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate syntax;

use log::debug;
use rustc::middle::cstore::ExternCrate;
use rustc_driver::Callbacks;
use rustc_interface::interface;
use semverver::run_analysis;
use std::{
    path::Path,
    process::{exit, Command},
};
use syntax::source_map::Pos;

/// Display semverver version.
fn show_version() {
    println!(env!("CARGO_PKG_VERSION"));
}

/// Main routine.
///
/// Find the sysroot before passing our args to the custom compiler driver we register.
fn main() {
    rustc_driver::init_rustc_env_logger();

    debug!("running rust-semverver compiler driver");
    exit(
        {
            use std::env;

            struct SemverCallbacks;

            impl Callbacks for SemverCallbacks {
                fn after_analysis(&mut self, compiler: &interface::Compiler) -> bool {
                    debug!("running rust-semverver after_analysis callback");

                    let verbose =
                        env::var("RUST_SEMVER_VERBOSE") == Ok("true".to_string());
                    let compact =
                        env::var("RUST_SEMVER_COMPACT") == Ok("true".to_string());
                    let api_guidelines =
                        env::var("RUST_SEMVER_API_GUIDELINES") == Ok("true".to_string());
                    let version = if let Ok(ver) = env::var("RUST_SEMVER_CRATE_VERSION") {
                        ver
                    } else {
                        "no_version".to_owned()
                    };

                    compiler.global_ctxt().unwrap().peek_mut().enter(|tcx| {
                        // To select the old and new crates we look at the position of the
                        // declaration in the source file. The first one will be the `old`
                        // and the other will be `new`. This is unfortunately a bit hacky...
                        // See issue #64 for details.

                        let mut crates: Vec<_> = tcx
                            .crates()
                            .iter()
                            .flat_map(|crate_num| {
                                let def_id = crate_num.as_def_id();

                                match tcx.extern_crate(def_id) {
                                    Some(ExternCrate {
                                        span, direct: true, ..
                                    }) if span.data().lo.to_usize() > 0 =>
                                        Some((span.data().lo.to_usize(), def_id)),
                                    _ => None,
                                }
                            })
                            .collect();

                        crates.sort_by_key(|&(span_lo, _)| span_lo);

                        if let [(_, old_def_id), (_, new_def_id)] = *crates.as_slice() {
                            debug!("running semver analysis");
                            let changes = run_analysis(tcx, old_def_id, new_def_id);
                            changes.output(tcx.sess, &version, verbose, compact, api_guidelines);
                        } else {
                            tcx.sess.err("could not find `old` and `new` crates");
                        }
                    });

                    debug!("rust-semverver after_analysis callback finished!");

                    false
                }
            }

            if env::args().any(|a| a == "--version" || a == "-V") {
                show_version();
                exit(0);
            }

            let sys_root = option_env!("SYSROOT")
                .map(String::from)
                .or_else(|| env::var("SYSROOT").ok())
                .or_else(|| {
                    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
                    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
                    home.and_then(|home| toolchain.map(|toolchain| format!("{}/toolchains/{}", home, toolchain)))
                })
                .or_else(|| {
                    Command::new("rustc")
                        .arg("--print")
                        .arg("sysroot")
                        .output()
                        .ok()
                        .and_then(|out| String::from_utf8(out.stdout).ok())
                        .map(|s| s.trim().to_owned())
                })
                .expect("need to specify SYSROOT env var during clippy compilation, or use rustup or multirust");

            // Setting RUSTC_WRAPPER causes Cargo to pass 'rustc' as the first argument.
            // We're invoking the compiler programmatically, so we ignore this/
            let mut orig_args: Vec<String> = env::args().collect();
            if orig_args.len() <= 1 {
                std::process::exit(1);
            }

            if Path::new(&orig_args[1]).file_stem() == Some("rustc".as_ref()) {
                // we still want to be able to invoke it normally though
                orig_args.remove(1);
            }

            // this conditional check for the --sysroot flag is there so users can call
            // `clippy_driver` directly
            // without having to pass --sysroot or anything
            let args: Vec<String> = if orig_args.iter().any(|s| s == "--sysroot") {
                orig_args.clone()
            } else {
                orig_args
                    .clone()
                    .into_iter()
                    .chain(Some("--sysroot".to_owned()))
                    .chain(Some(sys_root))
                    .collect()
            };

            let args = args;
            rustc_driver::run_compiler(&args, &mut SemverCallbacks, None, None)
        }
        .map_or_else(|_| 1, |_| 0),
    )
}
