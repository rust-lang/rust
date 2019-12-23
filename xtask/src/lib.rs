//! FIXME: write short doc here

pub mod codegen;

use anyhow::Context;
pub use anyhow::Result;
use std::{
    env, fs,
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use crate::codegen::Mode;

const TOOLCHAIN: &str = "stable";

pub fn project_root() -> PathBuf {
    Path::new(
        &env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
    )
    .ancestors()
    .nth(1)
    .unwrap()
    .to_path_buf()
}

pub struct Cmd<'a> {
    pub unix: &'a str,
    pub windows: &'a str,
    pub work_dir: &'a str,
}

impl Cmd<'_> {
    pub fn run(self) -> Result<()> {
        if cfg!(windows) {
            run(self.windows, self.work_dir)
        } else {
            run(self.unix, self.work_dir)
        }
    }
    pub fn run_with_output(self) -> Result<Output> {
        if cfg!(windows) {
            run_with_output(self.windows, self.work_dir)
        } else {
            run_with_output(self.unix, self.work_dir)
        }
    }
}

pub fn run(cmdline: &str, dir: &str) -> Result<()> {
    do_run(cmdline, dir, |c| {
        c.stdout(Stdio::inherit());
    })
    .map(|_| ())
}

pub fn run_with_output(cmdline: &str, dir: &str) -> Result<Output> {
    do_run(cmdline, dir, |_| {})
}

pub fn run_rustfmt(mode: Mode) -> Result<()> {
    match Command::new("rustup")
        .args(&["run", TOOLCHAIN, "--", "cargo", "fmt", "--version"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => install_rustfmt().context("install rustfmt")?,
    };

    if mode == Mode::Verify {
        run(&format!("rustup run {} -- cargo fmt -- --check", TOOLCHAIN), ".")?;
    } else {
        run(&format!("rustup run {} -- cargo fmt", TOOLCHAIN), ".")?;
    }
    Ok(())
}

pub fn install_rustfmt() -> Result<()> {
    run(&format!("rustup toolchain install {}", TOOLCHAIN), ".")?;
    run(&format!("rustup component add rustfmt --toolchain {}", TOOLCHAIN), ".")
}

pub fn install_pre_commit_hook() -> Result<()> {
    let result_path =
        PathBuf::from(format!("./.git/hooks/pre-commit{}", std::env::consts::EXE_SUFFIX));
    if !result_path.exists() {
        let me = std::env::current_exe()?;
        fs::copy(me, result_path)?;
    } else {
        Err(IoError::new(ErrorKind::AlreadyExists, "Git hook already created"))?;
    }
    Ok(())
}

pub fn run_clippy() -> Result<()> {
    match Command::new("rustup")
        .args(&["run", TOOLCHAIN, "--", "cargo", "clippy", "--version"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => install_clippy().context("install clippy")?,
    };

    let allowed_lints = [
        "clippy::collapsible_if",
        "clippy::map_clone", // FIXME: remove when Iterator::copied stabilizes (1.36.0)
        "clippy::needless_pass_by_value",
        "clippy::nonminimal_bool",
        "clippy::redundant_pattern_matching",
    ];
    run(
        &format!(
            "rustup run {} -- cargo clippy --all-features --all-targets -- -A {}",
            TOOLCHAIN,
            allowed_lints.join(" -A ")
        ),
        ".",
    )?;
    Ok(())
}

pub fn install_clippy() -> Result<()> {
    run(&format!("rustup toolchain install {}", TOOLCHAIN), ".")?;
    run(&format!("rustup component add clippy --toolchain {}", TOOLCHAIN), ".")
}

pub fn run_fuzzer() -> Result<()> {
    match Command::new("cargo")
        .args(&["fuzz", "--help"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => run("cargo install cargo-fuzz", ".")?,
    };

    run("rustup run nightly -- cargo fuzz run parser", "./crates/ra_syntax")
}

pub fn reformat_staged_files() -> Result<()> {
    run_rustfmt(Mode::Overwrite)?;
    let root = project_root();
    let output = Command::new("git")
        .arg("diff")
        .arg("--diff-filter=MAR")
        .arg("--name-only")
        .arg("--cached")
        .current_dir(&root)
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "`git diff --diff-filter=MAR --name-only --cached` exited with {}",
            output.status
        );
    }
    for line in String::from_utf8(output.stdout)?.lines() {
        run(&format!("git update-index --add {}", root.join(line).to_string_lossy()), ".")?;
    }
    Ok(())
}

fn do_run<F>(cmdline: &str, dir: &str, mut f: F) -> Result<Output>
where
    F: FnMut(&mut Command),
{
    eprintln!("\nwill run: {}", cmdline);
    let proj_dir = project_root().join(dir);
    let mut args = cmdline.split_whitespace();
    let exec = args.next().unwrap();
    let mut cmd = Command::new(exec);
    f(cmd.args(args).current_dir(proj_dir).stderr(Stdio::inherit()));
    let output = cmd.output().with_context(|| format!("running `{}`", cmdline))?;
    if !output.status.success() {
        anyhow::bail!("`{}` exited with {}", cmdline, output.status);
    }
    Ok(output)
}
