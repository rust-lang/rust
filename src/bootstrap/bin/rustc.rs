//! Shim which is passed to Cargo as "rustc" when running the bootstrap.
//!
//! This shim will take care of some various tasks that our build process
//! requires that Cargo can't quite do through normal configuration:
//!
//! 1. When compiling build scripts and build dependencies, we need a guaranteed
//!    full standard library available. The only compiler which actually has
//!    this is the snapshot, so we detect this situation and always compile with
//!    the snapshot compiler.
//! 2. We pass a bunch of `--cfg` and other flags based on what we're compiling
//!    (and this slightly differs based on a whether we're using a snapshot or
//!    not), so we do that all here.
//!
//! This may one day be replaced by RUSTFLAGS, but the dynamic nature of
//! switching compilers for the bootstrap and for build scripts will probably
//! never get replaced.

use std::env;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::time::Instant;

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();

    // Detect whether or not we're a build script depending on whether --target
    // is passed (a bit janky...)
    let target = args.windows(2).find(|w| &*w[0] == "--target").and_then(|w| w[1].to_str());
    let version = args.iter().find(|w| &**w == "-vV");

    let verbose = match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    };

    // Use a different compiler for build scripts, since there may not yet be a
    // libstd for the real compiler to use. However, if Cargo is attempting to
    // determine the version of the compiler, the real compiler needs to be
    // used. Currently, these two states are differentiated based on whether
    // --target and -vV is/isn't passed.
    let (rustc, libdir) = if target.is_none() && version.is_none() {
        ("RUSTC_SNAPSHOT", "RUSTC_SNAPSHOT_LIBDIR")
    } else {
        ("RUSTC_REAL", "RUSTC_LIBDIR")
    };
    let stage = env::var("RUSTC_STAGE").expect("RUSTC_STAGE was not set");
    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");
    let on_fail = env::var_os("RUSTC_ON_FAIL").map(Command::new);

    let rustc = env::var_os(rustc).unwrap_or_else(|| panic!("{:?} was not set", rustc));
    let libdir = env::var_os(libdir).unwrap_or_else(|| panic!("{:?} was not set", libdir));
    let mut dylib_path = bootstrap::util::dylib_path();
    dylib_path.insert(0, PathBuf::from(&libdir));

    let mut cmd = Command::new(rustc);
    cmd.args(&args).env(bootstrap::util::dylib_path_var(), env::join_paths(&dylib_path).unwrap());

    // Get the name of the crate we're compiling, if any.
    let crate_name =
        args.windows(2).find(|args| args[0] == "--crate-name").and_then(|args| args[1].to_str());

    if let Some(crate_name) = crate_name {
        if let Some(target) = env::var_os("RUSTC_TIME") {
            if target == "all"
                || target.into_string().unwrap().split(',').any(|c| c.trim() == crate_name)
            {
                cmd.arg("-Ztime");
            }
        }
    }

    // Print backtrace in case of ICE
    if env::var("RUSTC_BACKTRACE_ON_ICE").is_ok() && env::var("RUST_BACKTRACE").is_err() {
        cmd.env("RUST_BACKTRACE", "1");
    }

    if target.is_some() {
        // The stage0 compiler has a special sysroot distinct from what we
        // actually downloaded, so we just always pass the `--sysroot` option,
        // unless one is already set.
        if !args.iter().any(|arg| arg == "--sysroot") {
            cmd.arg("--sysroot").arg(&sysroot);
        }

        // If we're compiling specifically the `panic_abort` crate then we pass
        // the `-C panic=abort` option. Note that we do not do this for any
        // other crate intentionally as this is the only crate for now that we
        // ship with panic=abort.
        //
        // This... is a bit of a hack how we detect this. Ideally this
        // information should be encoded in the crate I guess? Would likely
        // require an RFC amendment to RFC 1513, however.
        //
        // `compiler_builtins` are unconditionally compiled with panic=abort to
        // workaround undefined references to `rust_eh_unwind_resume` generated
        // otherwise, see issue https://github.com/rust-lang/rust/issues/43095.
        if crate_name == Some("panic_abort")
            || crate_name == Some("compiler_builtins") && stage != "0"
        {
            cmd.arg("-C").arg("panic=abort");
        }

        // Set various options from config.toml to configure how we're building
        // code.
        let debug_assertions = match env::var("RUSTC_DEBUG_ASSERTIONS") {
            Ok(s) => {
                if s == "true" {
                    "y"
                } else {
                    "n"
                }
            }
            Err(..) => "n",
        };

        // The compiler builtins are pretty sensitive to symbols referenced in
        // libcore and such, so we never compile them with debug assertions.
        //
        // FIXME(rust-lang/cargo#7253) we should be doing this in `builder.rs`
        // with env vars instead of doing it here in this script.
        if crate_name == Some("compiler_builtins") {
            cmd.arg("-C").arg("debug-assertions=no");
        } else {
            cmd.arg("-C").arg(format!("debug-assertions={}", debug_assertions));
        }
    } else {
        // FIXME(rust-lang/cargo#5754) we shouldn't be using special env vars
        // here, but rather Cargo should know what flags to pass rustc itself.

        // Override linker if necessary.
        if let Ok(host_linker) = env::var("RUSTC_HOST_LINKER") {
            cmd.arg(format!("-Clinker={}", host_linker));
        }

        // Override linker flavor if necessary.
        if let Ok(host_linker_flavor) = env::var("RUSTC_HOST_LINKER_FLAVOR") {
            cmd.arg(format!("-Clinker-flavor={}", host_linker_flavor));
        }

        if let Ok(s) = env::var("RUSTC_HOST_CRT_STATIC") {
            if s == "true" {
                cmd.arg("-C").arg("target-feature=+crt-static");
            }
            if s == "false" {
                cmd.arg("-C").arg("target-feature=-crt-static");
            }
        }
    }

    if let Ok(map) = env::var("RUSTC_DEBUGINFO_MAP") {
        cmd.arg("--remap-path-prefix").arg(&map);
    }

    // Force all crates compiled by this compiler to (a) be unstable and (b)
    // allow the `rustc_private` feature to link to other unstable crates
    // also in the sysroot. We also do this for host crates, since those
    // may be proc macros, in which case we might ship them.
    if env::var_os("RUSTC_FORCE_UNSTABLE").is_some() && (stage != "0" || target.is_some()) {
        cmd.arg("-Z").arg("force-unstable-if-unmarked");
    }

    if verbose > 1 {
        eprintln!(
            "rustc command: {:?}={:?} {:?}",
            bootstrap::util::dylib_path_var(),
            env::join_paths(&dylib_path).unwrap(),
            cmd,
        );
        eprintln!("sysroot: {:?}", sysroot);
        eprintln!("libdir: {:?}", libdir);
    }

    if let Some(mut on_fail) = on_fail {
        let e = match cmd.status() {
            Ok(s) if s.success() => std::process::exit(0),
            e => e,
        };
        println!("\nDid not run successfully: {:?}\n{:?}\n-------------", e, cmd);
        exec_cmd(&mut on_fail).expect("could not run the backup command");
        std::process::exit(1);
    }

    if env::var_os("RUSTC_PRINT_STEP_TIMINGS").is_some() {
        if let Some(crate_name) = crate_name {
            let start = Instant::now();
            let status = cmd.status().unwrap_or_else(|_| panic!("\n\n failed to run {:?}", cmd));
            let dur = start.elapsed();

            let is_test = args.iter().any(|a| a == "--test");
            eprintln!(
                "[RUSTC-TIMING] {} test:{} {}.{:03}",
                crate_name,
                is_test,
                dur.as_secs(),
                dur.subsec_millis()
            );

            match status.code() {
                Some(i) => std::process::exit(i),
                None => {
                    eprintln!("rustc exited with {}", status);
                    std::process::exit(0xfe);
                }
            }
        }
    }

    let code = exec_cmd(&mut cmd).unwrap_or_else(|_| panic!("\n\n failed to run {:?}", cmd));
    std::process::exit(code);
}

#[cfg(unix)]
fn exec_cmd(cmd: &mut Command) -> io::Result<i32> {
    use std::os::unix::process::CommandExt;
    Err(cmd.exec())
}

#[cfg(not(unix))]
fn exec_cmd(cmd: &mut Command) -> io::Result<i32> {
    cmd.status().map(|status| status.code().unwrap())
}
