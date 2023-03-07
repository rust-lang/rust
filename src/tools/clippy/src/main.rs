#![cfg_attr(feature = "deny-warnings", deny(warnings))]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::env;
use std::path::PathBuf;
use std::process::{self, Command};

const CARGO_CLIPPY_HELP: &str = r#"Checks a package to catch common mistakes and improve your Rust code.

Usage:
    cargo clippy [options] [--] [<opts>...]

Common options:
    --no-deps                Run Clippy only on the given crate, without linting the dependencies
    --fix                    Automatically apply lint suggestions. This flag implies `--no-deps`
    -h, --help               Print this message
    -V, --version            Print version info and exit
    --explain LINT           Print the documentation for a given lint

Other options are the same as `cargo check`.

To allow or deny a lint from the command line you can use `cargo clippy --`
with:

    -W --warn OPT       Set lint warnings
    -A --allow OPT      Set lint allowed
    -D --deny OPT       Set lint denied
    -F --forbid OPT     Set lint forbidden

You can use tool lints to allow or deny lints from your code, e.g.:

    #[allow(clippy::needless_lifetimes)]
"#;

fn show_help() {
    println!("{CARGO_CLIPPY_HELP}");
}

fn show_version() {
    let version_info = rustc_tools_util::get_version_info!();
    println!("{version_info}");
}

pub fn main() {
    // Check for version and help flags even when invoked as 'cargo-clippy'
    if env::args().any(|a| a == "--help" || a == "-h") {
        show_help();
        return;
    }

    if env::args().any(|a| a == "--version" || a == "-V") {
        show_version();
        return;
    }

    if let Some(pos) = env::args().position(|a| a == "--explain") {
        if let Some(mut lint) = env::args().nth(pos + 1) {
            lint.make_ascii_lowercase();
            clippy_lints::explain(&lint.strip_prefix("clippy::").unwrap_or(&lint).replace('-', "_"));
        } else {
            show_help();
        }
        return;
    }

    if let Err(code) = process(env::args().skip(2)) {
        process::exit(code);
    }
}

struct ClippyCmd {
    cargo_subcommand: &'static str,
    args: Vec<String>,
    clippy_args: Vec<String>,
}

impl ClippyCmd {
    fn new<I>(mut old_args: I) -> Self
    where
        I: Iterator<Item = String>,
    {
        let mut cargo_subcommand = "check";
        let mut args = vec![];
        let mut clippy_args: Vec<String> = vec![];

        for arg in old_args.by_ref() {
            match arg.as_str() {
                "--fix" => {
                    cargo_subcommand = "fix";
                    continue;
                },
                "--no-deps" => {
                    clippy_args.push("--no-deps".into());
                    continue;
                },
                "--" => break,
                _ => {},
            }

            args.push(arg);
        }

        clippy_args.append(&mut (old_args.collect()));
        if cargo_subcommand == "fix" && !clippy_args.iter().any(|arg| arg == "--no-deps") {
            clippy_args.push("--no-deps".into());
        }

        Self {
            cargo_subcommand,
            args,
            clippy_args,
        }
    }

    fn path() -> PathBuf {
        let mut path = env::current_exe()
            .expect("current executable path invalid")
            .with_file_name("clippy-driver");

        if cfg!(windows) {
            path.set_extension("exe");
        }

        path
    }

    fn into_std_cmd(self) -> Command {
        let mut cmd = Command::new("cargo");
        let clippy_args: String = self
            .clippy_args
            .iter()
            .map(|arg| format!("{arg}__CLIPPY_HACKERY__"))
            .collect();

        // Currently, `CLIPPY_TERMINAL_WIDTH` is used only to format "unknown field" error messages.
        let terminal_width = termize::dimensions().map_or(0, |(w, _)| w);

        cmd.env("RUSTC_WORKSPACE_WRAPPER", Self::path())
            .env("CLIPPY_ARGS", clippy_args)
            .env("CLIPPY_TERMINAL_WIDTH", terminal_width.to_string())
            .arg(self.cargo_subcommand)
            .args(&self.args);

        cmd
    }
}

fn process<I>(old_args: I) -> Result<(), i32>
where
    I: Iterator<Item = String>,
{
    let cmd = ClippyCmd::new(old_args);

    let mut cmd = cmd.into_std_cmd();

    let exit_status = cmd
        .spawn()
        .expect("could not run cargo")
        .wait()
        .expect("failed to wait for cargo?");

    if exit_status.success() {
        Ok(())
    } else {
        Err(exit_status.code().unwrap_or(-1))
    }
}

#[cfg(test)]
mod tests {
    use super::ClippyCmd;

    #[test]
    fn fix() {
        let args = "cargo clippy --fix".split_whitespace().map(ToString::to_string);
        let cmd = ClippyCmd::new(args);
        assert_eq!("fix", cmd.cargo_subcommand);
        assert!(!cmd.args.iter().any(|arg| arg.ends_with("unstable-options")));
    }

    #[test]
    fn fix_implies_no_deps() {
        let args = "cargo clippy --fix".split_whitespace().map(ToString::to_string);
        let cmd = ClippyCmd::new(args);
        assert!(cmd.clippy_args.iter().any(|arg| arg == "--no-deps"));
    }

    #[test]
    fn no_deps_not_duplicated_with_fix() {
        let args = "cargo clippy --fix -- --no-deps"
            .split_whitespace()
            .map(ToString::to_string);
        let cmd = ClippyCmd::new(args);
        assert_eq!(cmd.clippy_args.iter().filter(|arg| *arg == "--no-deps").count(), 1);
    }

    #[test]
    fn check() {
        let args = "cargo clippy".split_whitespace().map(ToString::to_string);
        let cmd = ClippyCmd::new(args);
        assert_eq!("check", cmd.cargo_subcommand);
    }
}
