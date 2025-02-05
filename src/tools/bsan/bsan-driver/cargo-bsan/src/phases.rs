use rustc_version::VersionMeta;

use crate::arg::*;
use crate::setup::*;
use crate::util::*;
use crate::*;

#[derive(Clone, Debug)]
pub enum BSANCommand {
    /// Our own special 'setup' command.
    Setup,
    /// A command to be forwarded to cargo.
    Forward(String),
    /// Clean the cache
    Clean,
}

pub fn phase_cargo_bsan(mut args: impl Iterator<Item = String>) {
    if has_arg_flag("--help") || has_arg_flag("-h") {
        show_help();
        return;
    }
    if has_arg_flag("--version") || has_arg_flag("-V") {
        show_version();
        return;
    }

    let Some(subcommand) = args.next() else {
        show_error!(
            "`cargo bsan` needs to be called with a subcommand (e.g `run`, `test`, `clean`)"
        );
    };
    let subcommand = match &*subcommand {
        "setup" => BSANCommand::Setup,
        "test" | "t" | "run" | "r" | "nextest" => BSANCommand::Forward(subcommand),
        "clean" => BSANCommand::Clean,
        _ => show_error!(
            "`cargo bsan` supports the following subcommands: `run`, `test`, `nextest`, `clean`, and `setup`."
        ),
    };
    let verbose = num_arg_flag("-v");
    let quiet = has_arg_flag("-q") || has_arg_flag("--quiet");

    // Determine the involved architectures.
    let rustc_version = VersionMeta::for_command(bsan_for_host()).unwrap_or_else(|err| {
        panic!("failed to determine underlying rustc version of BSAN ({:?}):\n{err:?}", bsan())
    });

    let mut targets = get_arg_flag_values("--target").collect::<Vec<_>>();

    // If `targets` is empty, we need to add a `--target $HOST` flag ourselves, and also ensure
    // that the host target is indeed setup.
    let target_flag = if targets.is_empty() {
        let host = &rustc_version.host;
        targets.push(host.clone());
        Some(host)
    } else {
        show_error!("Cross-compilation is not supported at this time.");
    };

    // If cleaning the target directory & sysroot cache,
    // delete them then exit. There is no reason to setup a new
    // sysroot in this execution.
    if let BSANCommand::Clean = subcommand {
        clean_sysroot_dir();
        clean_target_dir();
        return;
    }

    for target in &targets {
        // We always setup.
        setup(&subcommand, target.as_str(), &rustc_version, verbose, quiet);
    }

    let bsan_sysroot = get_sysroot_dir();
    let bsan_path = find_bsan();

    let cargo_cmd = match subcommand {
        BSANCommand::Forward(s) => s,
        BSANCommand::Setup => return, // `cargo bsan setup` stops here.
        BSANCommand::Clean => unreachable!(),
    };

    let metadata = get_cargo_metadata();

    let mut cmd = cargo();
    cmd.arg(&cargo_cmd);

    // Set `--target-dir` to `bsan` inside the original target directory.
    let target_dir = get_target_dir(&metadata);
    cmd.arg("--target-dir").arg(target_dir);

    // In nextest we have to also forward the main `verb`.
    if cargo_cmd == "nextest" {
        cmd.arg(
            args.next()
                .unwrap_or_else(|| show_error!("`cargo bsan nextest` expects a verb (e.g. `run`)")),
        );
    }

    if let Some(target_flag) = target_flag {
        cmd.arg("--target");
        cmd.arg(target_flag);
    }

    if env::var_os("RUSTC_WRAPPER").is_some() {
        println!(
            "WARNING: Ignoring `RUSTC_WRAPPER` environment variable, BSAN does not support wrapping."
        );
    }
    cmd.args(args);

    if env::var_os("RUSTC_WRAPPER").is_some() {
        println!(
            "WARNING: Ignoring `RUSTC_WRAPPER` environment variable, BSAN does not support wrapping."
        );
    }
    cmd.env("RUSTC_WRAPPER", &bsan_path);

    // If both RUSTC_WORKSPACE_WRAPPER and RUSTC_WRAPPER are set,
    // then both are executed in succession. Providing an independent
    // workspace-level wrapper is not supported, so we clear this variable.
    if env::var_os("RUSTC_WORKSPACE_WRAPPER").is_some() {
        println!(
            "WARNING: Ignoring `RUSTC_WORKSPACE_WRAPPER` environment variable, BSAN does not support wrapping."
        );
    }
    cmd.env_remove("RUSTC_WORKSPACE_WRAPPER");

    // At this point, we've completed setup, so we have a sysroot.
    cmd.env("BSAN_SYSROOT", bsan_sysroot);
    if verbose > 0 {
        cmd.env("BSAN_VERBOSE", verbose.to_string()); // This makes the other phases verbose.
    }

    // Run cargo.
    debug_cmd("[cargo-bsan rustc]", verbose, &cmd);
    exec(cmd)
}
