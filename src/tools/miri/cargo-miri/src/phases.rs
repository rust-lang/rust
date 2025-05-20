//! Implements the various phases of `cargo miri run/test`.

use std::env;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;

use rustc_version::VersionMeta;

use crate::setup::*;
use crate::util::*;

const CARGO_MIRI_HELP: &str = r"Runs binary crates and tests in Miri

Usage:
    cargo miri [subcommand] [<cargo options>...] [--] [<program/test suite options>...]

Subcommands:
    run, r                   Run binaries
    test, t                  Run tests
    nextest                  Run tests with nextest (requires cargo-nextest installed)
    setup                    Only perform automatic setup, but without asking questions (for getting a proper libstd)
    clean                    Clean the Miri cache & target directory

The cargo options are exactly the same as for `cargo run` and `cargo test`, respectively.
Furthermore, the following environment variables are recognized for `run` and `test`:

    MIRIFLAGS                Extra flags to pass to the Miri driver. Use this to pass `-Zmiri-...` flags.

Examples:
    cargo miri run
    cargo miri test -- test-suite-filter

    cargo miri setup --print-sysroot
        This will print the path to the generated sysroot (and nothing else) on stdout.
        stderr will still contain progress information about how the build is doing.

";

fn show_help() {
    println!("{CARGO_MIRI_HELP}");
}

fn show_version() {
    print!("miri {}", env!("CARGO_PKG_VERSION"));
    let version = format!("{} {}", env!("GIT_HASH"), env!("COMMIT_DATE"));
    if version.len() > 1 {
        // If there is actually something here, print it.
        print!(" ({version})");
    }
    println!();
}

fn forward_patched_extern_arg(args: &mut impl Iterator<Item = String>, cmd: &mut Command) {
    cmd.arg("--extern"); // always forward flag, but adjust filename:
    let path = args.next().expect("`--extern` should be followed by a filename");
    if let Some(lib) = path.strip_suffix(".rlib") {
        // If this is an rlib, make it an rmeta.
        cmd.arg(format!("{lib}.rmeta"));
    } else {
        // Some other extern file (e.g. a `.so`). Forward unchanged.
        cmd.arg(path);
    }
}

pub fn phase_cargo_miri(mut args: impl Iterator<Item = String>) {
    // Check for version and help flags even when invoked as `cargo-miri`.
    if has_arg_flag("--help") || has_arg_flag("-h") {
        show_help();
        return;
    }
    if has_arg_flag("--version") || has_arg_flag("-V") {
        show_version();
        return;
    }

    // Require a subcommand before any flags.
    // We cannot know which of those flags take arguments and which do not,
    // so we cannot detect subcommands later.
    let Some(subcommand) = args.next() else {
        show_error!("`cargo miri` needs to be called with a subcommand (`run`, `test`, `clean`)");
    };
    let subcommand = match &*subcommand {
        "setup" => MiriCommand::Setup,
        "test" | "t" | "run" | "r" | "nextest" => MiriCommand::Forward(subcommand),
        "clean" => MiriCommand::Clean,
        _ =>
            show_error!(
                "`cargo miri` supports the following subcommands: `run`, `test`, `nextest`, `clean`, and `setup`."
            ),
    };
    let verbose = num_arg_flag("-v");
    let quiet = has_arg_flag("-q") || has_arg_flag("--quiet");

    // Determine the involved architectures.
    let rustc_version = VersionMeta::for_command(miri_for_host()).unwrap_or_else(|err| {
        panic!(
            "failed to determine underlying rustc version of Miri ({:?}):\n{err:?}",
            miri_for_host()
        )
    });
    let mut targets = get_arg_flag_values("--target").collect::<Vec<_>>();
    // If `targets` is empty, we need to add a `--target $HOST` flag ourselves, and also ensure
    // that the host target is indeed setup.
    let target_flag = if targets.is_empty() {
        let host = &rustc_version.host;
        targets.push(host.clone());
        Some(host)
    } else {
        // We don't need to add a `--target` flag, we just forward the user's flags.
        None
    };

    // If cleaning the target directory & sysroot cache,
    // delete them then exit. There is no reason to setup a new
    // sysroot in this execution.
    if let MiriCommand::Clean = subcommand {
        let metadata = get_cargo_metadata();
        clean_target_dir(&metadata);
        clean_sysroot();
        return;
    }

    for target in &targets {
        // We always setup.
        setup(&subcommand, target.as_str(), &rustc_version, verbose, quiet);
    }
    let miri_sysroot = get_sysroot_dir();

    // Invoke actual cargo for the job, but with different flags.
    // We re-use `cargo test` and `cargo run`, which makes target and binary handling very easy but
    // requires some extra work to make the build check-only (see all the `--emit` hacks below).
    // <https://github.com/rust-lang/miri/pull/1540#issuecomment-693553191> describes an alternative
    // approach that uses `cargo check`, making that part easier but target and binary handling
    // harder.
    let cargo_miri_path = env::current_exe()
        .expect("current executable path invalid")
        .into_os_string()
        .into_string()
        .expect("current executable path is not valid UTF-8");
    let cargo_cmd = match subcommand {
        MiriCommand::Forward(s) => s,
        MiriCommand::Setup => return, // `cargo miri setup` stops here.
        MiriCommand::Clean => unreachable!(),
    };
    let metadata = get_cargo_metadata();
    let mut cmd = cargo();
    cmd.arg(&cargo_cmd);
    // In nextest we have to also forward the main `verb`.
    if cargo_cmd == "nextest" {
        cmd.arg(
            args.next()
                .unwrap_or_else(|| show_error!("`cargo miri nextest` expects a verb (e.g. `run`)")),
        );
    }
    // We set the following flags *before* forwarding more arguments.
    // This is needed to fix <https://github.com/rust-lang/miri/issues/2829>: cargo will stop
    // interpreting things as flags when it sees the first positional argument.

    // Make sure the build target is explicitly set.
    // This is needed to make the `target.runner` settings do something,
    // and it later helps us detect which crates are proc-macro/build-script
    // (host crates) and which crates are needed for the program itself.
    if let Some(target_flag) = target_flag {
        cmd.arg("--target");
        cmd.arg(target_flag);
    }

    // Set ourselves as runner for al binaries invoked by cargo.
    // We use `all()` since `true` is not a thing in cfg-lang, but the empty conjunction is. :)
    let cargo_miri_path_for_toml = escape_for_toml(&cargo_miri_path);
    cmd.arg("--config")
        .arg(format!("target.'cfg(all())'.runner=[{cargo_miri_path_for_toml}, 'runner']"));

    // Set `--target-dir` to `miri` inside the original target directory.
    let target_dir = get_target_dir(&metadata);
    cmd.arg("--target-dir").arg(target_dir);
    // Only when running in x.py (where we are running with beta cargo): set `RUSTC_STAGE`.
    // Will have to be removed on next bootstrap bump. tag: cfg(bootstrap).
    if env::var_os("RUSTC_STAGE").is_some() {
        cmd.arg("-Zdoctest-xcompile");
    }

    // *After* we set all the flags that need setting, forward everything else. Make sure to skip
    // `--target-dir` (which would otherwise be set twice).
    for arg in
        ArgSplitFlagValue::from_string_iter(&mut args, "--target-dir").filter_map(Result::err)
    {
        if arg == "--many-seeds" || arg.starts_with("--many-seeds=") {
            show_error!(
                "ERROR: the `--many-seeds` flag has been removed from cargo-miri; use MIRIFLAGS=-Zmiri-many-seeds instead"
            );
        } else {
            cmd.arg(arg);
        }
    }
    // Forward all further arguments after `--` (not consumed by `ArgSplitFlagValue`) to cargo.
    cmd.args(args);

    // Set `RUSTC_WRAPPER` to ourselves.  Cargo will prepend that binary to its usual invocation,
    // i.e., the first argument is `rustc` -- which is what we use in `main` to distinguish
    // the two codepaths. (That extra argument is why we prefer this over setting `RUSTC`.)
    if env::var_os("RUSTC_WRAPPER").is_some() {
        println!(
            "WARNING: Ignoring `RUSTC_WRAPPER` environment variable, Miri does not support wrapping."
        );
    }
    cmd.env("RUSTC_WRAPPER", &cargo_miri_path);
    // There's also RUSTC_WORKSPACE_WRAPPER, which gets in the way of our own wrapping.
    if env::var_os("RUSTC_WORKSPACE_WRAPPER").is_some() {
        println!(
            "WARNING: Ignoring `RUSTC_WORKSPACE_WRAPPER` environment variable, Miri does not support wrapping."
        );
    }
    cmd.env_remove("RUSTC_WORKSPACE_WRAPPER");
    // We are going to invoke `MIRI` for everything, not `RUSTC`.
    if env::var_os("RUSTC").is_some() && env::var_os("MIRI").is_none() {
        println!(
            "WARNING: Ignoring `RUSTC` environment variable; set `MIRI` if you want to control the binary used as the driver."
        );
    }
    // Ideally we would set RUSTC to some non-existent path, so we can be sure our wrapping is
    // always applied. However, buggy build scripts (https://github.com/eyre-rs/eyre/issues/84) and
    // also cargo (https://github.com/rust-lang/cargo/issues/10885) will invoke `rustc` even when
    // `RUSTC_WRAPPER` is set, bypassing the wrapper. To make sure everything is coherent, we want
    // that to be the Miri driver, but acting as rustc, in host mode.
    //
    // In `main`, we need the value of `RUSTC` to distinguish RUSTC_WRAPPER invocations from rustdoc
    // or TARGET_RUNNER invocations, so we canonicalize it here to make it exceedingly unlikely that
    // there would be a collision with other invocations of cargo-miri (as rustdoc or as runner). We
    // explicitly do this even if RUSTC_STAGE is set, since for these builds we do *not* want the
    // bootstrap `rustc` thing in our way! Instead, we have MIRI_HOST_SYSROOT to use for host
    // builds.
    cmd.env("RUSTC", fs::canonicalize(find_miri()).unwrap());
    // In case we get invoked as RUSTC without the wrapper, let's be a host rustc. This makes no
    // sense for cross-interpretation situations, but without the wrapper, this will use the host
    // sysroot, so asking it to behave like a target build makes even less sense.
    cmd.env("MIRI_BE_RUSTC", "host"); // we better remember to *unset* this in the other phases!

    // Set rustdoc to us as well, so we can run doctests.
    if let Some(orig_rustdoc) = env::var_os("RUSTDOC") {
        cmd.env("MIRI_ORIG_RUSTDOC", orig_rustdoc);
    }
    cmd.env("RUSTDOC", &cargo_miri_path);

    // Forward some crucial information to our own re-invocations.
    cmd.env("MIRI_SYSROOT", miri_sysroot);
    cmd.env("MIRI_LOCAL_CRATES", local_crates(&metadata));
    if verbose > 0 {
        cmd.env("MIRI_VERBOSE", verbose.to_string()); // This makes the other phases verbose.
    }

    // Run cargo.
    debug_cmd("[cargo-miri cargo]", verbose, &cmd);
    exec(cmd)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum RustcPhase {
    /// `rustc` called during sysroot build.
    Setup,
    /// `rustc` called by `cargo` for regular build.
    Build,
    /// `rustc` called by `rustdoc` for doctest.
    Rustdoc,
}

pub fn phase_rustc(mut args: impl Iterator<Item = String>, phase: RustcPhase) {
    /// Determines if we are being invoked (as rustc) to build a crate for
    /// the "target" architecture, in contrast to the "host" architecture.
    /// Host crates are for build scripts and proc macros and still need to
    /// be built like normal; target crates need to be built for or interpreted
    /// by Miri.
    ///
    /// Currently, we detect this by checking for "--target=", which is
    /// never set for host crates. This matches what rustc bootstrap does,
    /// which hopefully makes it "reliable enough". This relies on us always
    /// invoking cargo itself with `--target`, which `in_cargo_miri` ensures.
    fn is_target_crate() -> bool {
        get_arg_flag_value("--target").is_some()
    }

    /// Returns whether or not Cargo invoked the wrapper (this binary) to compile
    /// the final, binary crate (either a test for 'cargo test', or a binary for 'cargo run')
    /// Cargo does not give us this information directly, so we need to check
    /// various command-line flags.
    fn is_runnable_crate() -> bool {
        // Determine whether this is cargo invoking rustc to get some infos. Ideally we'd check "is
        // there a filename passed to rustc", but that's very hard as we would have to know whether
        // e.g. `--print foo` is a booolean flag `--print` followed by filename `foo` or equivalent
        // to `--print=foo`. So instead we use this more fragile approach of detecting the presence
        // of a "query" flag rather than the absence of a filename.
        let info_query = get_arg_flag_value("--print").is_some() || has_arg_flag("-vV");
        if info_query {
            // Nothing to run.
            return false;
        }
        let is_bin = get_arg_flag_value("--crate-type").as_deref().unwrap_or("bin") == "bin";
        let is_test = has_arg_flag("--test");
        is_bin || is_test
    }

    fn out_filenames() -> Vec<PathBuf> {
        if let Some(out_file) = get_arg_flag_value("-o") {
            // `-o` has precedence over `--out-dir`.
            vec![PathBuf::from(out_file)]
        } else {
            let out_dir = get_arg_flag_value("--out-dir").unwrap_or_default();
            let path = PathBuf::from(out_dir);
            // Ask rustc for the filename (since that is target-dependent).
            let mut rustc = miri_for_host(); // sysroot doesn't matter for this so we just use the host
            rustc.arg("--print").arg("file-names");
            for flag in ["--crate-name", "--crate-type", "--target"] {
                for val in get_arg_flag_values(flag) {
                    rustc.arg(flag).arg(val);
                }
            }
            // This is technically passed as `-C extra-filename=...`, but the prefix seems unique
            // enough... (and cargo passes this before the filename so it should be unique)
            if let Some(extra) = get_arg_flag_value("extra-filename") {
                rustc.arg("-C").arg(format!("extra-filename={extra}"));
            }
            rustc.arg("-");

            let output = rustc.output().expect("cannot run rustc to determine file name");
            assert!(
                output.status.success(),
                "rustc failed when determining file name:\n{output:?}"
            );
            let output =
                String::from_utf8(output.stdout).expect("rustc returned non-UTF-8 filename");
            output.lines().filter(|l| !l.is_empty()).map(|l| path.join(l)).collect()
        }
    }

    let verbose = env::var("MIRI_VERBOSE")
        .map_or(0, |verbose| verbose.parse().expect("verbosity flag must be an integer"));
    let target_crate = is_target_crate();

    let store_json = |info: CrateRunInfo| {
        if get_arg_flag_value("--emit").unwrap_or_default().split(',').any(|e| e == "dep-info") {
            // Create a stub .d file to stop Cargo from "rebuilding" the crate:
            // https://github.com/rust-lang/miri/issues/1724#issuecomment-787115693
            // As we store a JSON file instead of building the crate here, an empty file is fine.
            let mut dep_info_name = PathBuf::from(get_arg_flag_value("--out-dir").unwrap());
            dep_info_name.push(format!(
                "{}{}.d",
                get_arg_flag_value("--crate-name").unwrap(),
                get_arg_flag_value("extra-filename").unwrap_or_default(),
            ));
            if verbose > 0 {
                eprintln!(
                    "[cargo-miri rustc] writing stub dep-info to `{}`",
                    dep_info_name.display()
                );
            }
            File::create(dep_info_name).expect("failed to create fake .d file");
        }

        for filename in out_filenames() {
            if verbose > 0 {
                eprintln!("[cargo-miri rustc] writing run info to `{}`", filename.display());
            }
            info.store(&filename);
        }
    };

    let runnable_crate = is_runnable_crate();

    if runnable_crate && target_crate {
        assert!(
            phase != RustcPhase::Setup,
            "there should be no interpretation during sysroot build"
        );
        let inside_rustdoc = phase == RustcPhase::Rustdoc;
        // This is the binary or test crate that we want to interpret under Miri.
        // But we cannot run it here, as cargo invoked us as a compiler -- our stdin and stdout are not
        // like we want them.
        // Instead of compiling, we write JSON into the output file with all the relevant command-line flags
        // and environment variables; this is used when cargo calls us again in the CARGO_TARGET_RUNNER phase.
        let env = CrateRunEnv::collect(args, inside_rustdoc);

        store_json(CrateRunInfo::RunWith(env.clone()));

        // Rustdoc expects us to exit with an error code if the test is marked as `compile_fail`,
        // just creating the JSON file is not enough: we need to detect syntax errors,
        // so we need to run Miri with `MIRI_BE_RUSTC` for a check-only build.
        if inside_rustdoc {
            let mut cmd = miri();

            // Ensure --emit argument for a check-only build is present.
            if let Some(val) =
                ArgFlagValueIter::from_str_iter(env.args.iter().map(|s| s as &str), "--emit").next()
            {
                // For `no_run` tests, rustdoc passes a `--emit` flag; make sure it has the right shape.
                assert_eq!(val, "metadata");
            } else {
                // For all other kinds of tests, we can just add our flag.
                cmd.arg("--emit=metadata");
            }

            // Alter the `-o` parameter so that it does not overwrite the JSON file we stored above.
            let mut args = env.args;
            for i in 0..args.len() {
                if args[i] == "-o" {
                    args[i + 1].push_str(".miri");
                }
            }

            cmd.args(&args);
            cmd.env("MIRI_BE_RUSTC", "target");

            if verbose > 0 {
                eprintln!(
                    "[cargo-miri rustc inside rustdoc] captured input:\n{}",
                    std::str::from_utf8(&env.stdin).unwrap()
                );
                eprintln!("[cargo-miri rustc inside rustdoc] going to run:\n{cmd:?}");
            }

            exec_with_pipe(cmd, &env.stdin);
        }

        return;
    }

    if runnable_crate && get_arg_flag_values("--extern").any(|krate| krate == "proc_macro") {
        // This is a "runnable" `proc-macro` crate (unit tests). We do not support
        // interpreting that under Miri now, so we write a JSON file to (display a
        // helpful message and) skip it in the runner phase.
        store_json(CrateRunInfo::SkipProcMacroTest);
        return;
    }

    let mut cmd = miri();
    let mut emit_link_hack = false;
    // Arguments are treated very differently depending on whether this crate is
    // for interpretation by Miri, or for use by a build script / proc macro.
    if target_crate {
        if phase != RustcPhase::Setup {
            // Set the sysroot -- except during setup, where we don't have an existing sysroot yet
            // and where the bootstrap wrapper adds its own `--sysroot` flag so we can't set ours.
            cmd.arg("--sysroot").arg(env::var_os("MIRI_SYSROOT").unwrap());
        }

        // Forward arguments, but patched.
        let emit_flag = "--emit";
        // This hack helps bootstrap run standard library tests in Miri. The issue is as follows:
        // when running `cargo miri test` on libcore, cargo builds a local copy of core and makes it
        // a dependency of the integration test crate. This copy duplicates all the lang items, so
        // the build fails. (Regular testing avoids this because the sysroot is a literal copy of
        // what `cargo build` produces, but since Miri builds its own sysroot this does not work for
        // us.) So we need to make it so that the locally built libcore contains all the items from
        // `core`, but does not re-define them -- we want to replace the entire crate but a
        // re-export of the sysroot crate. We do this by swapping out the source file: if
        // `MIRI_REPLACE_LIBRS_IF_NOT_TEST` is set and we are building a `lib.rs` file, and a
        // `lib.miri.rs` file exists in the same folder, we build that instead. But crucially we
        // only do that for the library, not the unit test crate (which would be runnable) or
        // rustdoc (which would have a different `phase`).
        let replace_librs = env::var_os("MIRI_REPLACE_LIBRS_IF_NOT_TEST").is_some()
            && !runnable_crate
            && phase == RustcPhase::Build;
        while let Some(arg) = args.next() {
            // Patch `--emit`: remove "link" from "--emit" to make this a check-only build.
            if let Some(val) = arg.strip_prefix(emit_flag) {
                // Patch this argument. First, extract its value.
                let val =
                    val.strip_prefix('=').expect("`cargo` should pass `--emit=X` as one argument");
                let mut val: Vec<_> = val.split(',').collect();
                // Now make sure "link" is not in there, but "metadata" is.
                if let Some(i) = val.iter().position(|&s| s == "link") {
                    emit_link_hack = true;
                    val.remove(i);
                    if !val.contains(&"metadata") {
                        val.push("metadata");
                    }
                }
                cmd.arg(format!("{emit_flag}={}", val.join(",")));
                continue;
            }
            // Patch `--extern` filenames, since Cargo sometimes passes stub `.rlib` files:
            // https://github.com/rust-lang/miri/issues/1705
            if arg == "--extern" {
                forward_patched_extern_arg(&mut args, &mut cmd);
                continue;
            }
            // If the REPLACE_LIBRS hack is enabled and we are building a `lib.rs` file, and a
            // `lib.miri.rs` file exists, then build that instead.
            if replace_librs {
                let path = Path::new(&arg);
                if path.file_name().is_some_and(|f| f == "lib.rs") && path.is_file() {
                    let miri_rs = Path::new(&arg).with_extension("miri.rs");
                    if miri_rs.is_file() {
                        if verbose > 0 {
                            eprintln!("Performing REPLACE_LIBRS hack: {arg:?} -> {miri_rs:?}");
                        }
                        cmd.arg(miri_rs);
                        continue;
                    }
                }
            }
            // Fallback: just propagate the argument.
            cmd.arg(arg);
        }

        // During setup, patch the panic runtime for `libpanic_abort` (mirroring what bootstrap usually does).
        if phase == RustcPhase::Setup
            && get_arg_flag_value("--crate-name").as_deref() == Some("panic_abort")
        {
            cmd.arg("-C").arg("panic=abort");
        }
    } else {
        // This is a host crate.
        // When we're running `cargo-miri` from `x.py` we need to pass the sysroot explicitly
        // due to bootstrap complications.
        if let Some(sysroot) = env::var_os("MIRI_HOST_SYSROOT") {
            cmd.arg("--sysroot").arg(sysroot);
        }

        // Forward everything.
        cmd.args(args);
    }

    // We want to compile, not interpret. We still use Miri to make sure the compiler version etc
    // are the exact same as what is used for interpretation.
    // MIRI_DEFAULT_ARGS should not be used to build host crates, hence setting "target" or "host"
    // as the value here to help Miri differentiate them.
    cmd.env("MIRI_BE_RUSTC", if target_crate { "target" } else { "host" });

    // Run it.
    if verbose > 0 {
        eprintln!("[cargo-miri rustc] target_crate={target_crate} runnable_crate={runnable_crate}");
    }

    // Create a stub .rlib file if "link" was requested by cargo.
    // This is necessary to prevent cargo from doing rebuilds all the time.
    if emit_link_hack {
        for filename in out_filenames() {
            if verbose > 0 {
                eprintln!("[cargo-miri rustc] creating fake lib file at `{}`", filename.display());
            }
            File::create(filename).expect("failed to create fake lib file");
        }
    }

    debug_cmd("[cargo-miri rustc]", verbose, &cmd);
    exec(cmd);
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum RunnerPhase {
    /// `cargo` is running a binary
    Cargo,
    /// `rustdoc` is running a binary
    Rustdoc,
}

pub fn phase_runner(mut binary_args: impl Iterator<Item = String>, phase: RunnerPhase) {
    let verbose = env::var("MIRI_VERBOSE")
        .map_or(0, |verbose| verbose.parse().expect("verbosity flag must be an integer"));

    let binary = binary_args.next().unwrap();
    let file = File::open(&binary)
        .unwrap_or_else(|_| show_error!(
            "file {:?} not found or `cargo-miri` invoked incorrectly; please only invoke this binary through `cargo miri`", binary
        ));
    let file = BufReader::new(file);
    let binary_args = binary_args.collect::<Vec<_>>();

    let info = serde_json::from_reader(file).unwrap_or_else(|_| {
        show_error!("file {:?} contains outdated or invalid JSON; try `cargo clean`", binary)
    });
    let info = match info {
        CrateRunInfo::RunWith(info) => info,
        CrateRunInfo::SkipProcMacroTest => {
            eprintln!(
                "Running unit tests of `proc-macro` crates is not currently supported by Miri."
            );
            return;
        }
    };

    let mut cmd = miri();

    // Set missing env vars. We prefer build-time env vars over run-time ones; see
    // <https://github.com/rust-lang/miri/issues/1661> for the kind of issue that fixes.
    for (name, val) in &info.env {
        // `CARGO_MAKEFLAGS` contains information about how to reach the jobserver, but by the time
        // the program is being run, that jobserver no longer exists (cargo only runs the jobserver
        // for the build portion of `cargo run`/`cargo test`). Hence we shouldn't forward this.
        // Also see <https://github.com/rust-lang/rust/pull/113730>.
        if name == "CARGO_MAKEFLAGS" {
            continue;
        }
        if let Some(old_val) = env::var_os(name) {
            if *old_val == *val {
                // This one did not actually change, no need to re-set it.
                // (This keeps the `debug_cmd` below more manageable.)
                continue;
            } else if verbose > 0 {
                eprintln!(
                    "[cargo-miri runner] Overwriting run-time env var {name:?}={old_val:?} with build-time value {val:?}"
                );
            }
        }
        cmd.env(name, val);
    }

    if phase != RunnerPhase::Rustdoc {
        // Set the sysroot. Not necessary in rustdoc, where we already set the sysroot in
        // `phase_rustdoc`. rustdoc will forward that flag when invoking rustc (i.e., us), so the
        // flag is present in `info.args`.
        cmd.arg("--sysroot").arg(env::var_os("MIRI_SYSROOT").unwrap());
    }
    // Forward rustc arguments.
    // We need to patch "--extern" filenames because we forced a check-only
    // build without cargo knowing about that: replace `.rlib` suffix by
    // `.rmeta`.
    // We also need to remove `--error-format` as cargo specifies that to be JSON,
    // but when we run here, cargo does not interpret the JSON any more. `--json`
    // then also needs to be dropped.
    let mut args = info.args.iter();
    while let Some(arg) = args.next() {
        if arg == "--extern" {
            forward_patched_extern_arg(&mut (&mut args).cloned(), &mut cmd);
        } else if let Some(suffix) = arg.strip_prefix("--error-format") {
            assert!(suffix.starts_with('='));
            // Drop this argument.
        } else if let Some(suffix) = arg.strip_prefix("--json") {
            assert!(suffix.starts_with('='));
            // Drop this argument.
        } else {
            cmd.arg(arg);
        }
    }
    // Respect `MIRIFLAGS`.
    if let Ok(a) = env::var("MIRIFLAGS") {
        let args = flagsplit(&a);
        cmd.args(args);
    }

    // Then pass binary arguments.
    cmd.arg("--");
    cmd.args(&binary_args);

    // Make sure we use the build-time working directory for interpreting Miri/rustc arguments.
    // But then we need to switch to the run-time one, which we instruct Miri to do by setting `MIRI_CWD`.
    cmd.current_dir(&info.current_dir);
    cmd.env("MIRI_CWD", env::current_dir().unwrap());

    // Run it.
    debug_cmd("[cargo-miri runner]", verbose, &cmd);

    match phase {
        RunnerPhase::Rustdoc => exec_with_pipe(cmd, &info.stdin),
        RunnerPhase::Cargo => exec(cmd),
    }
}

pub fn phase_rustdoc(mut args: impl Iterator<Item = String>) {
    let verbose = env::var("MIRI_VERBOSE")
        .map_or(0, |verbose| verbose.parse().expect("verbosity flag must be an integer"));

    // phase_cargo_miri sets the RUSTDOC env var to ourselves, and puts a backup
    // of the old value into MIRI_ORIG_RUSTDOC. So that's what we have to invoke now.
    let rustdoc = env::var("MIRI_ORIG_RUSTDOC").unwrap_or("rustdoc".to_string());
    let mut cmd = Command::new(rustdoc);

    while let Some(arg) = args.next() {
        if arg == "--extern" {
            // Patch --extern arguments to use *.rmeta files, since phase_cargo_rustc only creates stub *.rlib files.
            forward_patched_extern_arg(&mut args, &mut cmd);
        } else {
            cmd.arg(arg);
        }
    }

    // Doctests of `proc-macro` crates (and their dependencies) are always built for the host,
    // so we are not able to run them in Miri.
    if get_arg_flag_values("--crate-type").any(|crate_type| crate_type == "proc-macro") {
        eprintln!("Running doctests of `proc-macro` crates is not currently supported by Miri.");
        return;
    }

    // For each doctest, rustdoc starts two child processes: first the test is compiled,
    // then the produced executable is invoked. We want to reroute both of these to cargo-miri,
    // such that the first time we'll enter phase_cargo_rustc, and phase_cargo_runner second.
    //
    // rustdoc invokes the test-builder by forwarding most of its own arguments, which makes
    // it difficult to determine when phase_cargo_rustc should run instead of phase_cargo_rustdoc.
    // Furthermore, the test code is passed via stdin, rather than a temporary file, so we need
    // to let phase_cargo_rustc know to expect that. We'll use this environment variable as a flag:
    cmd.env("MIRI_CALLED_FROM_RUSTDOC", "1");

    // The `--test-builder` is an unstable rustdoc features,
    // which is disabled by default. We first need to enable them explicitly:
    cmd.arg("-Zunstable-options");

    // rustdoc needs to know the right sysroot.
    cmd.arg("--sysroot").arg(env::var_os("MIRI_SYSROOT").unwrap());
    // make sure the 'miri' flag is set for rustdoc
    cmd.arg("--cfg").arg("miri");

    // Make rustdoc call us back for the build.
    // (cargo already sets `--test-runtool` to us since we are the cargo test runner.)
    let cargo_miri_path = env::current_exe().expect("current executable path invalid");
    cmd.arg("--test-builder").arg(&cargo_miri_path); // invoked by forwarding most arguments

    debug_cmd("[cargo-miri rustdoc]", verbose, &cmd);
    exec(cmd)
}
