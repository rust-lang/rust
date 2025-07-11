//! Run bootstrap from any subdirectory of a rust compiler checkout.
//!
//! We prefer `exec`, to avoid adding an extra process in the process tree.
//! However, since `exec` isn't available on Windows, we indirect through
//! `exec_or_status`, which will call `exec` on unix and `status` on Windows.
//!
//! We use `powershell.exe x.ps1` on Windows, and `sh -c x` on Unix, those are
//! the ones that call `x.py`. We use `sh -c` on Unix, because it is a standard.
//! We also don't use `pwsh` on Windows, because it is not installed by default;

use std::env::consts::EXE_EXTENSION;
use std::env::{self};
use std::io;
use std::path::Path;
use std::process::{self, Command, ExitStatus};

const PYTHON: &str = "python";
const PYTHON2: &str = "python2";
const PYTHON3: &str = "python3";

fn python() -> &'static str {
    let Some(path) = env::var_os("PATH") else {
        return PYTHON;
    };

    let mut python2 = false;
    let mut python3 = false;

    for dir in env::split_paths(&path) {
        // `python` should always take precedence over python2 / python3 if it exists
        if dir.join(PYTHON).with_extension(EXE_EXTENSION).exists() {
            return PYTHON;
        }

        python2 |= dir.join(PYTHON2).with_extension(EXE_EXTENSION).exists();
        python3 |= dir.join(PYTHON3).with_extension(EXE_EXTENSION).exists();
    }

    // try 3 before 2
    if python3 {
        PYTHON3
    } else if python2 {
        PYTHON2
    } else {
        // Python was not found on path, so exit
        eprintln!("Unable to find python in your PATH. Please check it is installed.");
        process::exit(1);
    }
}

#[cfg(windows)]
fn x_command(dir: &Path) -> Command {
    let mut cmd = Command::new("powershell.exe");
    cmd.args([
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "RemoteSigned",
        "-Command",
        "./x.ps1",
    ])
    .current_dir(dir);
    cmd
}

#[cfg(unix)]
fn x_command(dir: &Path) -> Command {
    Command::new(dir.join("x"))
}

#[cfg(not(any(windows, unix)))]
fn x_command(_dir: &Path) -> Command {
    compile_error!("Unsupported platform");
}

#[cfg(unix)]
fn exec_or_status(command: &mut Command) -> io::Result<ExitStatus> {
    use std::os::unix::process::CommandExt;
    Err(command.exec())
}

#[cfg(not(unix))]
fn exec_or_status(command: &mut Command) -> io::Result<ExitStatus> {
    command.status()
}

fn handle_result(result: io::Result<ExitStatus>, cmd: Command) {
    match result {
        Err(error) => {
            eprintln!("Failed to invoke `{cmd:?}`: {error}");
        }
        Ok(status) => {
            process::exit(status.code().unwrap_or(1));
        }
    }
}

fn main() {
    if env::args().nth(1).is_some_and(|s| s == "--wrapper-version") {
        let version = env!("CARGO_PKG_VERSION");
        println!("{version}");
        return;
    }

    let current = match env::current_dir() {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("Failed to get current directory: {err}");
            process::exit(1);
        }
    };
    for dir in current.ancestors() {
        let candidate = dir.join("x.py");
        if candidate.exists() {
            let shell_script_candidate = dir.join("x");
            let mut cmd: Command;
            if shell_script_candidate.exists() {
                cmd = x_command(dir);
                cmd.args(env::args().skip(1)).current_dir(dir);
            } else {
                // For older checkouts that do not have the x shell script, default to python
                cmd = Command::new(python());
                cmd.arg(&candidate).args(env::args().skip(1)).current_dir(dir);
            }
            let result = exec_or_status(&mut cmd);
            handle_result(result, cmd);
        }
    }

    eprintln!(
        "x.py not found. Please run inside of a checkout of `https://github.com/rust-lang/rust`."
    );

    process::exit(1);
}
