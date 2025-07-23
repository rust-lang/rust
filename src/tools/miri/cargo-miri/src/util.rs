use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::ops::Not;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::{Metadata, MetadataCommand};
use serde::{Deserialize, Serialize};

pub use crate::arg::*;

pub fn show_error_(msg: &impl std::fmt::Display) -> ! {
    eprintln!("fatal error: {msg}");
    std::process::exit(1)
}

macro_rules! show_error {
    ($($tt:tt)*) => { crate::util::show_error_(&format_args!($($tt)*)) };
}
pub(crate) use show_error;

/// The information to run a crate with the given environment.
#[derive(Clone, Serialize, Deserialize)]
pub struct CrateRunEnv {
    /// The command-line arguments.
    pub args: Vec<String>,
    /// The environment.
    pub env: Vec<(OsString, OsString)>,
    /// The current working directory.
    pub current_dir: OsString,
    /// The contents passed via standard input.
    pub stdin: Vec<u8>,
}

impl CrateRunEnv {
    /// Gather all the information we need.
    pub fn collect(args: impl Iterator<Item = String>, capture_stdin: bool) -> Self {
        let args = args.collect();
        let env = env::vars_os().collect();
        let current_dir = env::current_dir().unwrap().into_os_string();

        let mut stdin = Vec::new();
        if capture_stdin {
            std::io::stdin().lock().read_to_end(&mut stdin).expect("cannot read stdin");
        }

        CrateRunEnv { args, env, current_dir, stdin }
    }
}

/// The information Miri needs to run a crate. Stored as JSON when the crate is "compiled".
#[derive(Serialize, Deserialize)]
pub enum CrateRunInfo {
    /// Run it with the given environment.
    RunWith(CrateRunEnv),
    /// Skip it as Miri does not support interpreting such kind of crates.
    SkipProcMacroTest,
}

impl CrateRunInfo {
    pub fn store(&self, filename: &Path) {
        let file = File::create(filename)
            .unwrap_or_else(|_| show_error!("cannot create `{}`", filename.display()));
        let file = BufWriter::new(file);
        serde_json::ser::to_writer(file, self)
            .unwrap_or_else(|_| show_error!("cannot write to `{}`", filename.display()));
    }
}

#[derive(Clone, Debug)]
pub enum MiriCommand {
    /// Our own special 'setup' command.
    Setup,
    /// A command to be forwarded to cargo.
    Forward(String),
    /// Clean the miri cache
    Clean,
}

/// Escapes `s` in a way that is suitable for using it as a string literal in TOML syntax.
pub fn escape_for_toml(s: &str) -> String {
    // We want to surround this string in quotes `"`. So we first escape all quotes,
    // and also all backslashes (that are used to escape quotes).
    let s = s.replace('\\', r"\\").replace('"', r#"\""#);
    format!("\"{s}\"")
}

/// Returns the path to the `miri` binary
pub fn find_miri() -> PathBuf {
    if let Some(path) = env::var_os("MIRI") {
        return path.into();
    }
    // Assume it is in the same directory as ourselves.
    let mut path = std::env::current_exe().expect("current executable path invalid");
    path.set_file_name(format!("miri{}", env::consts::EXE_SUFFIX));
    path
}

pub fn miri() -> Command {
    let mut cmd = Command::new(find_miri());
    // We never want to inherit this from the environment.
    // However, this is sometimes set in the environment to work around build scripts that don't
    // honor RUSTC_WRAPPER. So remove it again in case it is set.
    cmd.env_remove("MIRI_BE_RUSTC");
    cmd
}

pub fn miri_for_host() -> Command {
    let mut cmd = miri();
    cmd.env("MIRI_BE_RUSTC", "host");
    cmd
}

pub fn cargo() -> Command {
    Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
}

pub fn flagsplit(flags: &str) -> Vec<String> {
    // This code is taken from `RUSTFLAGS` handling in cargo.
    flags.split(' ').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

/// Execute the `Command`, where possible by replacing the current process with a new process
/// described by the `Command`. Then exit this process with the exit code of the new process.
pub fn exec(mut cmd: Command) -> ! {
    // On non-Unix imitate POSIX exec as closely as we can
    #[cfg(not(unix))]
    {
        let exit_status =
            cmd.status().unwrap_or_else(|err| panic!("failed to run `{cmd:?}`:\n{err}"));
        std::process::exit(exit_status.code().unwrap_or(-1))
    }
    // On Unix targets, actually exec.
    // If exec returns, process setup has failed. This is the same error condition as the expect in
    // the non-Unix case.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        panic!("failed to run `{cmd:?}`:\n{err}")
    }
}

/// Execute the `Command`, then exit this process with the exit code of the new process.
/// `input` is also piped to the new process's stdin.
pub fn exec_with_pipe(mut cmd: Command, input: &[u8]) -> ! {
    // We can't use `exec` since then the background thread will stop running.
    cmd.stdin(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("failed to spawn process");
    let child_stdin = child.stdin.take().unwrap();
    // Write stdin in a background thread, as it may block.
    let exit_status = std::thread::scope(|s| {
        s.spawn(|| {
            let mut child_stdin = child_stdin;
            // Ignore failure, it is most likely due to the process having terminated.
            let _ = child_stdin.write_all(input);
        });
        child.wait().expect("failed to run command")
    });
    std::process::exit(exit_status.code().unwrap_or(-1))
}

pub fn ask_to_run(mut cmd: Command, ask: bool, text: &str) {
    // Disable interactive prompts in CI (GitHub Actions, Travis, AppVeyor, etc).
    // Azure doesn't set `CI` though (nothing to see here, just Microsoft being Microsoft),
    // so we also check their `TF_BUILD`.
    let is_ci = env::var_os("CI").is_some() || env::var_os("TF_BUILD").is_some();
    if ask && !is_ci {
        let mut buf = String::new();
        print!("I will run `{cmd:?}` to {text}. Proceed? [Y/n] ");
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut buf).unwrap();
        match buf.trim().to_lowercase().as_ref() {
            // Proceed.
            "" | "y" | "yes" => {}
            "n" | "no" => show_error!("aborting as per your request"),
            a => show_error!("invalid answer `{}`", a),
        };
    } else {
        eprintln!("Running `{cmd:?}` to {text}.");
    }

    if cmd.status().unwrap_or_else(|_| panic!("failed to execute {cmd:?}")).success().not() {
        show_error!("failed to {}", text);
    }
}

// Computes the extra flags that need to be passed to cargo to make it behave like the current
// cargo invocation.
fn cargo_extra_flags() -> Vec<String> {
    let mut flags = Vec::new();
    // Forward `--config` flags.
    let config_flag = "--config";
    for arg in get_arg_flag_values(config_flag) {
        flags.push(config_flag.to_string());
        flags.push(arg);
    }

    // Forward `--manifest-path`.
    let manifest_flag = "--manifest-path";
    if let Some(manifest) = get_arg_flag_value(manifest_flag) {
        flags.push(manifest_flag.to_string());
        flags.push(manifest);
    }

    // Forwarding `--target-dir` would make sense, but `cargo metadata` does not support that flag.

    flags
}

pub fn get_cargo_metadata() -> Metadata {
    // This will honor the `CARGO` env var the same way our `cargo()` does.
    MetadataCommand::new().no_deps().other_options(cargo_extra_flags()).exec().unwrap()
}

/// Pulls all the crates in this workspace from the cargo metadata.
/// Additionally, somewhere between cargo metadata and TyCtxt, '-' gets replaced with '_' so we
/// make that same transformation here.
pub fn local_crates(metadata: &Metadata) -> String {
    assert!(!metadata.workspace_members.is_empty());
    let package_name_by_id: HashMap<_, _> =
        metadata.packages.iter().map(|package| (&package.id, package.name.as_str())).collect();
    metadata
        .workspace_members
        .iter()
        .map(|id| package_name_by_id[id].replace('-', "_"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Debug-print a command that is going to be run.
pub fn debug_cmd(prefix: &str, verbose: usize, cmd: &Command) {
    if verbose == 0 {
        return;
    }
    eprintln!("{prefix} running command: {cmd:?}");
}

/// Get the target directory for miri output.
///
/// Either in an argument passed-in, or from cargo metadata.
pub fn get_target_dir(meta: &Metadata) -> PathBuf {
    let mut output = match get_arg_flag_value("--target-dir") {
        Some(dir) => PathBuf::from(dir),
        None => meta.target_directory.clone().into_std_path_buf(),
    };
    output.push("miri");
    output
}

/// Determines where the sysroot of this execution is
///
/// Either in a user-specified spot by an envar, or in a default cache location.
pub fn get_sysroot_dir() -> PathBuf {
    match std::env::var_os("MIRI_SYSROOT") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let user_dirs = directories::ProjectDirs::from("org", "rust-lang", "miri").unwrap();
            user_dirs.cache_dir().to_owned()
        }
    }
}

/// An idempotent version of the stdlib's remove_dir_all
/// it is considered a success if the directory was not there.
fn remove_dir_all_idem(dir: &Path) -> std::io::Result<()> {
    match std::fs::remove_dir_all(dir) {
        Ok(_) => Ok(()),
        // If the directory doesn't exist, it is still a success.
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

/// Deletes the Miri sysroot cache
/// Returns an error if the MIRI_SYSROOT env var is set.
pub fn clean_sysroot() {
    if std::env::var_os("MIRI_SYSROOT").is_some() {
        show_error!(
            "MIRI_SYSROOT is set. Please clean your custom sysroot cache directory manually."
        )
    }

    let sysroot_dir = get_sysroot_dir();

    eprintln!("Cleaning sysroot cache at {}", sysroot_dir.display());

    // Keep it simple, just remove the directory.
    remove_dir_all_idem(&sysroot_dir).unwrap_or_else(|err| show_error!("{}", err));
}

/// Deletes the Miri target directory
pub fn clean_target_dir(meta: &Metadata) {
    let target_dir = get_target_dir(meta);

    eprintln!("Cleaning target directory at {}", target_dir.display());

    remove_dir_all_idem(&target_dir).unwrap_or_else(|err| show_error!("{}", err))
}
