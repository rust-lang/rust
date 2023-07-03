use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::ops::Not;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::{Metadata, MetadataCommand};
use serde::{Deserialize, Serialize};

pub use crate::arg::*;

pub fn show_error(msg: &impl std::fmt::Display) -> ! {
    eprintln!("fatal error: {msg}");
    std::process::exit(1)
}

macro_rules! show_error {
    ($($tt:tt)*) => { crate::util::show_error(&format_args!($($tt)*)) };
}

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
    let mut path = std::env::current_exe().expect("current executable path invalid");
    if cfg!(windows) {
        path.set_file_name("miri.exe");
    } else {
        path.set_file_name("miri");
    }
    path
}

pub fn miri() -> Command {
    Command::new(find_miri())
}

pub fn miri_for_host() -> Command {
    let mut cmd = miri();
    cmd.env("MIRI_BE_RUSTC", "host");
    cmd
}

pub fn cargo() -> Command {
    Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
}

/// Execute the `Command`, where possible by replacing the current process with a new process
/// described by the `Command`. Then exit this process with the exit code of the new process.
pub fn exec(mut cmd: Command) -> ! {
    // On non-Unix imitate POSIX exec as closely as we can
    #[cfg(not(unix))]
    {
        let exit_status = cmd.status().expect("failed to run command");
        std::process::exit(exit_status.code().unwrap_or(-1))
    }
    // On Unix targets, actually exec.
    // If exec returns, process setup has failed. This is the same error condition as the expect in
    // the non-Unix case.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let error = cmd.exec();
        panic!("failed to run command: {error}")
    }
}

/// Execute the `Command`, where possible by replacing the current process with a new process
/// described by the `Command`. Then exit this process with the exit code of the new process.
/// `input` is also piped to the new process's stdin, on cfg(unix) platforms by writing its
/// contents to `path` first, then setting stdin to that file.
pub fn exec_with_pipe<P>(mut cmd: Command, input: &[u8], path: P) -> !
where
    P: AsRef<Path>,
{
    #[cfg(unix)]
    {
        // Write the bytes we want to send to stdin out to a file
        std::fs::write(&path, input).unwrap();
        // Open the file for reading, and set our new stdin to it
        let stdin = File::open(&path).unwrap();
        cmd.stdin(stdin);
        // Unlink the file so that it is fully cleaned up as soon as the new process exits
        std::fs::remove_file(&path).unwrap();
        // Finally, we can hand off control.
        exec(cmd)
    }
    #[cfg(not(unix))]
    {
        drop(path); // We don't need the path, we can pipe the bytes directly
        cmd.stdin(std::process::Stdio::piped());
        let mut child = cmd.spawn().expect("failed to spawn process");
        {
            let stdin = child.stdin.as_mut().expect("failed to open stdin");
            stdin.write_all(input).expect("failed to write out test source");
        }
        let exit_status = child.wait().expect("failed to run command");
        std::process::exit(exit_status.code().unwrap_or(-1))
    }
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
    // `-Zunstable-options` is required by `--config`.
    flags.push("-Zunstable-options".to_string());

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
/// Workspace members are emitted like "miri 0.1.0 (path+file:///path/to/miri)"
/// Additionally, somewhere between cargo metadata and TyCtxt, '-' gets replaced with '_' so we
/// make that same transformation here.
pub fn local_crates(metadata: &Metadata) -> String {
    assert!(!metadata.workspace_members.is_empty());
    let mut local_crates = String::new();
    for member in &metadata.workspace_members {
        let name = member.repr.split(' ').next().unwrap();
        let name = name.replace('-', "_");
        local_crates.push_str(&name);
        local_crates.push(',');
    }
    local_crates.pop(); // Remove the trailing ','

    local_crates
}

fn env_vars_from_cmd(cmd: &Command) -> Vec<(String, String)> {
    let mut envs = HashMap::new();
    for (key, value) in std::env::vars() {
        envs.insert(key, value);
    }
    for (key, value) in cmd.get_envs() {
        if let Some(value) = value {
            envs.insert(key.to_string_lossy().to_string(), value.to_string_lossy().to_string());
        } else {
            envs.remove(&key.to_string_lossy().to_string());
        }
    }
    let mut envs: Vec<_> = envs.into_iter().collect();
    envs.sort();
    envs
}

/// Debug-print a command that is going to be run.
pub fn debug_cmd(prefix: &str, verbose: usize, cmd: &Command) {
    if verbose == 0 {
        return;
    }
    // We only do a single `eprintln!` call to minimize concurrency interactions.
    let mut out = prefix.to_string();
    writeln!(out, " running command: env \\").unwrap();
    if verbose > 1 {
        // Print the full environment this will be called in.
        for (key, value) in env_vars_from_cmd(cmd) {
            writeln!(out, "{key}={value:?} \\").unwrap();
        }
    } else {
        // Print only what has been changed for this `cmd`.
        for (var, val) in cmd.get_envs() {
            if let Some(val) = val {
                writeln!(out, "{}={val:?} \\", var.to_string_lossy()).unwrap();
            } else {
                writeln!(out, "--unset={}", var.to_string_lossy()).unwrap();
            }
        }
    }
    write!(out, "{cmd:?}").unwrap();
    eprintln!("{out}");
}
