//! Shim which is passed to Cargo as "rustdoc" when running the bootstrap.
//!
//! See comments in `src/bootstrap/rustc.rs` for more information.

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{exit, Command};

include!("../dylib_util.rs");

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let stage = env::var("RUSTC_STAGE").unwrap_or_else(|_| {
        // Don't panic here; it's reasonable to try and run these shims directly. Give a helpful error instead.
        eprintln!("rustc shim: fatal: RUSTC_STAGE was not set");
        eprintln!("rustc shim: note: use `x.py build -vvv` to see all environment variables set by bootstrap");
        exit(101);
    });
    let rustdoc = env::var_os("RUSTDOC_REAL").expect("RUSTDOC_REAL was not set");
    let libdir = env::var_os("RUSTDOC_LIBDIR").expect("RUSTDOC_LIBDIR was not set");
    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");

    // Detect whether or not we're a build script depending on whether --target
    // is passed (a bit janky...)
    let target = args.windows(2).find(|w| &*w[0] == "--target").and_then(|w| w[1].to_str());

    use std::str::FromStr;

    let verbose = match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    };

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
    if let Some(linker) = env::var_os("RUSTDOC_LINKER") {
        let mut arg = OsString::from("-Clinker=");
        arg.push(&linker);
        cmd.arg(arg);
    }
    if let Ok(no_threads) = env::var("RUSTDOC_LLD_NO_THREADS") {
        cmd.arg("-Clink-arg=-fuse-ld=lld");
        cmd.arg(format!("-Clink-arg=-Wl,{}", no_threads));
    }
    // Cargo doesn't pass RUSTDOCFLAGS to proc_macros:
    // https://github.com/rust-lang/cargo/issues/4423
    // Thus, if we are on stage 0, we explicitly set `--cfg=bootstrap`.
    // We also declare that the flag is expected, which we need to do to not
    // get warnings about it being unexpected.
    if stage == "0" {
        cmd.arg("--cfg=bootstrap");
    }
    cmd.arg("-Zunstable-options");
    cmd.arg("--check-cfg=values(bootstrap)");

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
