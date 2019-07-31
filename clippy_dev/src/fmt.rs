use shell_escape::escape;
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use walkdir::WalkDir;

#[derive(Debug)]
pub enum CliError {
    CommandFailed(String),
    IoError(io::Error),
    ProjectRootNotFound,
    WalkDirError(walkdir::Error),
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

pub fn run(check: bool, verbose: bool) {
    fn try_run(context: &FmtContext) -> Result<bool, CliError> {
        let mut success = true;

        let project_root = project_root()?;

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
            CliError::ProjectRootNotFound => {
                eprintln!("error: Can't determine root of project. Please run inside a Clippy working dir.");
            },
            CliError::WalkDirError(err) => {
                eprintln!("error: {}", err);
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
            eprintln!("Run `./util/dev fmt` to update formatting.");
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
    let arg_display: Vec<_> = args
        .iter()
        .map(|a| escape(a.as_ref().to_string_lossy()).to_owned())
        .collect();

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

fn project_root() -> Result<PathBuf, CliError> {
    let current_dir = std::env::current_dir()?;
    for path in current_dir.ancestors() {
        let result = std::fs::read_to_string(path.join("Cargo.toml"));
        if let Err(err) = &result {
            if err.kind() == io::ErrorKind::NotFound {
                continue;
            }
        }

        let content = result?;
        if content.contains("[package]\nname = \"clippy\"") {
            return Ok(path.to_path_buf());
        }
    }

    Err(CliError::ProjectRootNotFound)
}
