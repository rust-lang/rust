//! Shim which is passed to Cargo as "rustdoc" when running the bootstrap.
//!
//! See comments in `src/bootstrap/rustc.rs` for more information.

#![deny(warnings)]

extern crate bootstrap;

use std::env;
use std::process::Command;
use std::path::PathBuf;

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let rustdoc = env::var_os("RUSTDOC_REAL").expect("RUSTDOC_REAL was not set");
    let libdir = env::var_os("RUSTDOC_LIBDIR").expect("RUSTDOC_LIBDIR was not set");
    let stage = env::var("RUSTC_STAGE").expect("RUSTC_STAGE was not set");
    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");

    use std::str::FromStr;

    let verbose = match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    };

    let mut dylib_path = bootstrap::util::dylib_path();
    dylib_path.insert(0, PathBuf::from(libdir.clone()));

    //FIXME(misdreavus): once stdsimd uses cfg(rustdoc) instead of cfg(dox), remove the `--cfg dox`
    //arguments here
    let mut cmd = Command::new(rustdoc);
    cmd.args(&args)
        .arg("--cfg")
        .arg(format!("stage{}", stage))
        .arg("--cfg")
        .arg("dox")
        .arg("--sysroot")
        .arg(sysroot)
        .env(bootstrap::util::dylib_path_var(),
             env::join_paths(&dylib_path).unwrap());

    // Force all crates compiled by this compiler to (a) be unstable and (b)
    // allow the `rustc_private` feature to link to other unstable crates
    // also in the sysroot.
    if env::var_os("RUSTC_FORCE_UNSTABLE").is_some() {
        cmd.arg("-Z").arg("force-unstable-if-unmarked");
    }
    if let Some(linker) = env::var_os("RUSTC_TARGET_LINKER") {
        cmd.arg("--linker").arg(linker).arg("-Z").arg("unstable-options");
    }

    // Bootstrap's Cargo-command builder sets this variable to the current Rust version; let's pick
    // it up so we can make rustdoc print this into the docs
    if let Some(version) = env::var_os("RUSTDOC_CRATE_VERSION") {
        // This "unstable-options" can be removed when `--crate-version` is stabilized
        cmd.arg("-Z")
           .arg("unstable-options")
           .arg("--crate-version").arg(version);
    }

    if verbose > 1 {
        eprintln!("rustdoc command: {:?}", cmd);
        eprintln!("libdir: {:?}", libdir);
    }

    std::process::exit(match cmd.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => panic!("\n\nfailed to run {:?}: {}\n\n", cmd, e),
    })
}
