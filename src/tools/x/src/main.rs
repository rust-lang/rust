//! Run `x.py` from any subdirectory of a rust compiler checkout.
//!
//! We prefer `exec`, to avoid adding an extra process in the process tree.
//! However, since `exec` isn't available on Windows, we indirect through
//! `exec_or_status`, which will call `exec` on unix and `status` on Windows.
//!
//! We use `python`, `python3`, or `python2` as the python interpreter to run
//! `x.py`, in that order of preference.

use std::{
    env, io,
    process::{self, Command, ExitStatus},
};

const PYTHON: &str = "python";
const PYTHON2: &str = "python2";
const PYTHON3: &str = "python3";

fn python() -> &'static str {
    let val = match env::var_os("PATH") {
        Some(val) => val,
        None => return PYTHON,
    };

    let mut python2 = false;
    let mut python3 = false;

    for dir in env::split_paths(&val) {
        if dir.join(PYTHON).exists() {
            return PYTHON;
        }

        python2 |= dir.join(PYTHON2).exists();
        python3 |= dir.join(PYTHON3).exists();
    }

    if python3 {
        PYTHON3
    } else if python2 {
        PYTHON2
    } else {
        PYTHON
    }
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

fn main() {
    let current = match env::current_dir() {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("Failed to get current directory: {}", err);
            process::exit(1);
        }
    };

    for dir in current.ancestors() {
        let candidate = dir.join("x.py");

        if candidate.exists() {
            let mut python = Command::new(python());

            python.arg(&candidate).args(env::args().skip(1)).current_dir(dir);

            let result = exec_or_status(&mut python);

            match result {
                Err(error) => {
                    eprintln!("Failed to invoke `{}`: {}", candidate.display(), error);
                }
                Ok(status) => {
                    process::exit(status.code().unwrap_or(1));
                }
            }
        }
    }

    eprintln!(
        "x.py not found. Please run inside of a checkout of `https://github.com/rust-lang/rust`."
    );

    process::exit(1);
}
