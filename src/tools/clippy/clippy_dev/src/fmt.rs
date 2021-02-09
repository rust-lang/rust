use crate::clippy_project_root;
use shell_escape::escape;
use std::ffi::OsStr;
use std::path::Path;
use std::process::{self, Command};
use std::{fs, io};
use walkdir::WalkDir;

#[derive(Debug)]
pub enum CliError {
    CommandFailed(String),
    IoError(io::Error),
    RustfmtNotInstalled,
    WalkDirError(walkdir::Error),
    RaSetupActive,
}

impl From<io::Error> for CliError {
    fn from(error: io::Error) -> Self {
        Self::IoError(error)
    }
}

impl From<walkdir::Error> for CliError {
    fn from(error: walkdir::Error) -> Self {
        Self::WalkDirError(error)
    }
}

struct FmtContext {
    check: bool,
    verbose: bool,
}

// the "main" function of cargo dev fmt
pub fn run(check: bool, verbose: bool) {
    fn try_run(context: &FmtContext) -> Result<bool, CliError> {
        let mut success = true;

        let project_root = clippy_project_root();

        // if we added a local rustc repo as path dependency to clippy for rust analyzer, we do NOT want to
        // format because rustfmt would also format the entire rustc repo as it is a local
        // dependency
        if fs::read_to_string(project_root.join("Cargo.toml"))
            .expect("Failed to read clippy Cargo.toml")
            .contains(&"[target.'cfg(NOT_A_PLATFORM)'.dependencies]")
        {
            return Err(CliError::RaSetupActive);
        }

        rustfmt_test(context)?;

        success &= cargo_fmt(context, project_root.as_path())?;
        success &= cargo_fmt(context, &project_root.join("clippy_dev"))?;
        success &= cargo_fmt(context, &project_root.join("rustc_tools_util"))?;

        for entry in WalkDir::new(project_root.join("tests")) {
            let entry = entry?;
            let path = entry.path();

            if path.extension() != Some("rs".as_ref())
                || entry.file_name() == "ice-3891.rs"
                // Avoid rustfmt bug rust-lang/rustfmt#1873
                || cfg!(windows) && entry.file_name() == "implicit_hasher.rs"
            {
                continue;
            }

            success &= rustfmt(context, &path)?;
        }

        Ok(success)
    }

    fn output_err(err: CliError) {
        match err {
            CliError::CommandFailed(command) => {
                eprintln!("error: A command failed! `{}`", command);
            },
            CliError::IoError(err) => {
                eprintln!("error: {}", err);
            },
            CliError::RustfmtNotInstalled => {
                eprintln!("error: rustfmt nightly is not installed.");
            },
            CliError::WalkDirError(err) => {
                eprintln!("error: {}", err);
            },
            CliError::RaSetupActive => {
                eprintln!(
                    "error: a local rustc repo is enabled as path dependency via `cargo dev ra_setup`.
Not formatting because that would format the local repo as well!
Please revert the changes to Cargo.tomls first."
                );
            },
        }
    }

    let context = FmtContext { check, verbose };
    let result = try_run(&context);
    let code = match result {
        Ok(true) => 0,
        Ok(false) => {
            eprintln!();
            eprintln!("Formatting check failed.");
            eprintln!("Run `cargo dev fmt` to update formatting.");
            1
        },
        Err(err) => {
            output_err(err);
            1
        },
    };
    process::exit(code);
}

fn format_command(program: impl AsRef<OsStr>, dir: impl AsRef<Path>, args: &[impl AsRef<OsStr>]) -> String {
    let arg_display: Vec<_> = args.iter().map(|a| escape(a.as_ref().to_string_lossy())).collect();

    format!(
        "cd {} && {} {}",
        escape(dir.as_ref().to_string_lossy()),
        escape(program.as_ref().to_string_lossy()),
        arg_display.join(" ")
    )
}

fn exec(
    context: &FmtContext,
    program: impl AsRef<OsStr>,
    dir: impl AsRef<Path>,
    args: &[impl AsRef<OsStr>],
) -> Result<bool, CliError> {
    if context.verbose {
        println!("{}", format_command(&program, &dir, args));
    }

    let mut child = Command::new(&program).current_dir(&dir).args(args.iter()).spawn()?;
    let code = child.wait()?;
    let success = code.success();

    if !context.check && !success {
        return Err(CliError::CommandFailed(format_command(&program, &dir, args)));
    }

    Ok(success)
}

fn cargo_fmt(context: &FmtContext, path: &Path) -> Result<bool, CliError> {
    let mut args = vec!["+nightly", "fmt", "--all"];
    if context.check {
        args.push("--");
        args.push("--check");
    }
    let success = exec(context, "cargo", path, &args)?;

    Ok(success)
}

fn rustfmt_test(context: &FmtContext) -> Result<(), CliError> {
    let program = "rustfmt";
    let dir = std::env::current_dir()?;
    let args = &["+nightly", "--version"];

    if context.verbose {
        println!("{}", format_command(&program, &dir, args));
    }

    let output = Command::new(&program).current_dir(&dir).args(args.iter()).output()?;

    if output.status.success() {
        Ok(())
    } else if std::str::from_utf8(&output.stderr)
        .unwrap_or("")
        .starts_with("error: 'rustfmt' is not installed")
    {
        Err(CliError::RustfmtNotInstalled)
    } else {
        Err(CliError::CommandFailed(format_command(&program, &dir, args)))
    }
}

fn rustfmt(context: &FmtContext, path: &Path) -> Result<bool, CliError> {
    let mut args = vec!["+nightly".as_ref(), path.as_os_str()];
    if context.check {
        args.push("--check".as_ref());
    }
    let success = exec(context, "rustfmt", std::env::current_dir()?, &args)?;
    if !success {
        eprintln!("rustfmt failed on {}", path.display());
    }
    Ok(success)
}
