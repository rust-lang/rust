//! Shim which is passed to Cargo as "rustdoc" when running the bootstrap.
//!
//! See comments in `src/bootstrap/rustc.rs` for more information.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use shared_helpers::{
    dylib_path, dylib_path_var, maybe_dump, parse_rustc_stage, parse_rustc_verbose,
    parse_value_from_args,
};

#[path = "../utils/shared_helpers.rs"]
mod shared_helpers;

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();

    let stage = parse_rustc_stage();
    let verbose = parse_rustc_verbose();

    let rustdoc = env::var_os("RUSTDOC_REAL").expect("RUSTDOC_REAL was not set");
    let libdir = env::var_os("RUSTDOC_LIBDIR").expect("RUSTDOC_LIBDIR was not set");
    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");

    // Detect whether or not we're a build script depending on whether --target
    // is passed (a bit janky...)
    let target = parse_value_from_args(&args, "--target");

    let mut dylib_path = dylib_path();
    dylib_path.insert(0, PathBuf::from(libdir.clone()));

    let mut cmd = Command::new(rustdoc);

    if target.is_some() {
        // The stage0 compiler has a special sysroot distinct from what we
        // actually downloaded, so we just always pass the `--sysroot` option,
        // unless one is already set.
        if !args.iter().any(|arg| arg == "--sysroot") {
            cmd.arg("--sysroot").arg(&sysroot);
        }
    }

    cmd.args(&args);
    cmd.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

    // Force all crates compiled by this compiler to (a) be unstable and (b)
    // allow the `rustc_private` feature to link to other unstable crates
    // also in the sysroot.
    if env::var_os("RUSTC_FORCE_UNSTABLE").is_some() {
        cmd.arg("-Z").arg("force-unstable-if-unmarked");
    }
    // Cargo doesn't pass RUSTDOCFLAGS to proc_macros:
    // https://github.com/rust-lang/cargo/issues/4423
    // Thus, if we are on stage 0, we explicitly set `--cfg=bootstrap`.
    // We also declare that the flag is expected, which we need to do to not
    // get warnings about it being unexpected.
    if stage == 0 {
        cmd.arg("--cfg=bootstrap");
    }

    maybe_dump(format!("stage{}-rustdoc", stage + 1), &cmd);

    if verbose > 1 {
        eprintln!(
            "rustdoc command: {:?}={:?} {:?}",
            dylib_path_var(),
            env::join_paths(&dylib_path).unwrap(),
            cmd,
        );
        eprintln!("sysroot: {sysroot:?}");
        eprintln!("libdir: {libdir:?}");
    }

    std::process::exit(match cmd.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => panic!("\n\nfailed to run {cmd:?}: {e}\n\n"),
    })
}
