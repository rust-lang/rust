//! Shim which is passed to Cargo as "rustdoc" when running the bootstrap.
//!
//! See comments in `src/bootstrap/rustc.rs` for more information.

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

include!("../dylib_util.rs");

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let rustdoc = env::var_os("RUSTDOC_REAL").expect("RUSTDOC_REAL was not set");
    let libdir = env::var_os("RUSTDOC_LIBDIR").expect("RUSTDOC_LIBDIR was not set");
    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");

    use std::str::FromStr;

    let verbose = match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    };

    let mut dylib_path = dylib_path();
    dylib_path.insert(0, PathBuf::from(libdir.clone()));

    let mut cmd = Command::new(rustdoc);
    cmd.args(&args)
        .arg("--sysroot")
        .arg(&sysroot)
        .env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

    // Force all crates compiled by this compiler to (a) be unstable and (b)
    // allow the `rustc_private` feature to link to other unstable crates
    // also in the sysroot.
    if env::var_os("RUSTC_FORCE_UNSTABLE").is_some() {
        cmd.arg("-Z").arg("force-unstable-if-unmarked");
    }
    if let Some(linker) = env::var_os("RUSTDOC_LINKER") {
        let mut arg = OsString::from("-Clinker=");
        arg.push(&linker);
        cmd.arg(arg);
    }
    if env::var_os("RUSTDOC_FUSE_LD_LLD").is_some() {
        cmd.arg("-Clink-arg=-fuse-ld=lld");
        if cfg!(windows) {
            cmd.arg("-Clink-arg=-Wl,/threads:1");
        } else {
            cmd.arg("-Clink-arg=-Wl,--threads=1");
        }
    }

    // Needed to be able to run all rustdoc tests.
    if let Some(ref x) = env::var_os("RUSTDOC_RESOURCE_SUFFIX") {
        // This "unstable-options" can be removed when `--resource-suffix` is stabilized
        cmd.arg("-Z").arg("unstable-options");
        cmd.arg("--resource-suffix").arg(x);
    }

    if verbose > 1 {
        eprintln!(
            "rustdoc command: {:?}={:?} {:?}",
            dylib_path_var(),
            env::join_paths(&dylib_path).unwrap(),
            cmd,
        );
        eprintln!("sysroot: {:?}", sysroot);
        eprintln!("libdir: {:?}", libdir);
    }

    std::process::exit(match cmd.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => panic!("\n\nfailed to run {:?}: {}\n\n", cmd, e),
    })
}
