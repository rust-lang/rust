// Copyright 2015-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Inspired by Paul Woolcock's cargo-fmt (https://github.com/pwoolcoc/cargo-fmt/)

#![cfg(not(test))]
#![deny(warnings)]

extern crate cargo_metadata;
extern crate getopts;
extern crate serde_json as json;

use std::collections::HashSet;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::iter::FromIterator;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::str;

use getopts::{Matches, Options};

fn main() {
    let exit_status = execute();
    std::io::stdout().flush().unwrap();
    std::process::exit(exit_status);
}

const SUCCESS: i32 = 0;
const FAILURE: i32 = 1;

fn execute() -> i32 {
    let mut opts = getopts::Options::new();
    opts.optflag("h", "help", "show this message");
    opts.optflag("q", "quiet", "no output printed to stdout");
    opts.optflag("v", "verbose", "use verbose output");
    opts.optmulti(
        "p",
        "package",
        "specify package to format (only usable in workspaces)",
        "<package>",
    );
    opts.optflag("", "version", "print rustfmt version and exit");
    opts.optflag("", "all", "format all packages (only usable in workspaces)");

    // If there is any invalid argument passed to `cargo fmt`, return without formatting.
    let mut is_package_arg = false;
    for arg in env::args().skip(2).take_while(|a| a != "--") {
        if arg.starts_with('-') {
            is_package_arg = arg.starts_with("--package");
        } else if !is_package_arg {
            print_usage_to_stderr(&opts, &format!("Invalid argument: `{}`.", arg));
            return FAILURE;
        } else {
            is_package_arg = false;
        }
    }

    let matches = match opts.parse(env::args().skip(1).take_while(|a| a != "--")) {
        Ok(m) => m,
        Err(e) => {
            print_usage_to_stderr(&opts, &e.to_string());
            return FAILURE;
        }
    };

    let verbosity = match (matches.opt_present("v"), matches.opt_present("q")) {
        (false, false) => Verbosity::Normal,
        (false, true) => Verbosity::Quiet,
        (true, false) => Verbosity::Verbose,
        (true, true) => {
            print_usage_to_stderr(&opts, "quiet mode and verbose mode are not compatible");
            return FAILURE;
        }
    };

    if matches.opt_present("h") {
        print_usage_to_stdout(&opts, "");
        return SUCCESS;
    }

    if matches.opt_present("version") {
        return handle_command_status(get_version(verbosity), &opts);
    }

    let strategy = CargoFmtStrategy::from_matches(&matches);
    handle_command_status(format_crate(verbosity, &strategy), &opts)
}

macro_rules! print_usage {
    ($print:ident, $opts:ident, $reason:expr) => {{
        let msg = format!("{}\nusage: cargo fmt [options]", $reason);
        $print!(
            "{}\nThis utility formats all bin and lib files of the current crate using rustfmt. \
             Arguments after `--` are passed to rustfmt.",
            $opts.usage(&msg)
        );
    }};
}

fn print_usage_to_stdout(opts: &Options, reason: &str) {
    print_usage!(println, opts, reason);
}

fn print_usage_to_stderr(opts: &Options, reason: &str) {
    print_usage!(eprintln, opts, reason);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verbosity {
    Verbose,
    Normal,
    Quiet,
}

fn handle_command_status(status: Result<ExitStatus, io::Error>, opts: &getopts::Options) -> i32 {
    match status {
        Err(e) => {
            print_usage_to_stderr(opts, &e.to_string());
            FAILURE
        }
        Ok(status) => {
            if status.success() {
                SUCCESS
            } else {
                status.code().unwrap_or(FAILURE)
            }
        }
    }
}

fn get_version(verbosity: Verbosity) -> Result<ExitStatus, io::Error> {
    run_rustfmt(&[], &[String::from("--version")], verbosity)
}

fn format_crate(
    verbosity: Verbosity,
    strategy: &CargoFmtStrategy,
) -> Result<ExitStatus, io::Error> {
    let rustfmt_args = get_fmt_args();
    let targets = if rustfmt_args
        .iter()
        .any(|s| ["--print-config", "-h", "--help", "-V", "--verison"].contains(&s.as_str()))
    {
        HashSet::new()
    } else {
        get_targets(strategy)?
    };

    // Currently only bin and lib files get formatted
    let files: Vec<_> = targets
        .into_iter()
        .inspect(|t| {
            if verbosity == Verbosity::Verbose {
                println!("[{}] {:?}", t.kind, t.path)
            }
        }).map(|t| t.path)
        .collect();

    run_rustfmt(&files, &rustfmt_args, verbosity)
}

fn get_fmt_args() -> Vec<String> {
    // All arguments after -- are passed to rustfmt
    env::args().skip_while(|a| a != "--").skip(1).collect()
}

/// Target uses a `path` field for equality and hashing.
#[derive(Debug)]
pub struct Target {
    /// A path to the main source file of the target.
    path: PathBuf,
    /// A kind of target (e.g. lib, bin, example, ...).
    kind: String,
}

impl Target {
    pub fn from_target(target: &cargo_metadata::Target) -> Self {
        let path = PathBuf::from(&target.src_path);
        let canonicalized = fs::canonicalize(&path).unwrap_or(path);

        Target {
            path: canonicalized,
            kind: target.kind[0].clone(),
        }
    }
}

impl PartialEq for Target {
    fn eq(&self, other: &Target) -> bool {
        self.path == other.path
    }
}

impl Eq for Target {}

impl Hash for Target {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CargoFmtStrategy {
    /// Format every packages and dependencies.
    All,
    /// Format packages that are specified by the command line argument.
    Some(Vec<String>),
    /// Format the root packages only.
    Root,
}

impl CargoFmtStrategy {
    pub fn from_matches(matches: &Matches) -> CargoFmtStrategy {
        match (matches.opt_present("all"), matches.opt_present("p")) {
            (false, false) => CargoFmtStrategy::Root,
            (true, _) => CargoFmtStrategy::All,
            (false, true) => CargoFmtStrategy::Some(matches.opt_strs("p")),
        }
    }
}

/// Based on the specified `CargoFmtStrategy`, returns a set of main source files.
fn get_targets(strategy: &CargoFmtStrategy) -> Result<HashSet<Target>, io::Error> {
    let mut targets = HashSet::new();

    match *strategy {
        CargoFmtStrategy::Root => get_targets_root_only(&mut targets)?,
        CargoFmtStrategy::All => get_targets_recursive(None, &mut targets, &mut HashSet::new())?,
        CargoFmtStrategy::Some(ref hitlist) => get_targets_with_hitlist(hitlist, &mut targets)?,
    }

    if targets.is_empty() {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to find targets".to_owned(),
        ))
    } else {
        Ok(targets)
    }
}

fn get_targets_root_only(targets: &mut HashSet<Target>) -> Result<(), io::Error> {
    let metadata = get_cargo_metadata(None)?;
    let current_dir = env::current_dir()?.canonicalize()?;
    let current_dir_manifest = current_dir.join("Cargo.toml");
    let workspace_root_path = PathBuf::from(&metadata.workspace_root).canonicalize()?;
    let in_workspace_root = workspace_root_path == current_dir;

    for package in metadata.packages {
        if in_workspace_root || PathBuf::from(&package.manifest_path) == current_dir_manifest {
            for target in package.targets {
                targets.insert(Target::from_target(&target));
            }
        }
    }

    Ok(())
}

fn get_targets_recursive(
    manifest_path: Option<&Path>,
    mut targets: &mut HashSet<Target>,
    visited: &mut HashSet<String>,
) -> Result<(), io::Error> {
    let metadata = get_cargo_metadata(manifest_path)?;

    for package in metadata.packages {
        add_targets(&package.targets, &mut targets);

        // Look for local dependencies.
        for dependency in package.dependencies {
            if dependency.source.is_some() || visited.contains(&dependency.name) {
                continue;
            }

            let mut manifest_path = PathBuf::from(&package.manifest_path);

            manifest_path.pop();
            manifest_path.push(&dependency.name);
            manifest_path.push("Cargo.toml");

            if manifest_path.exists() {
                visited.insert(dependency.name);
                get_targets_recursive(Some(&manifest_path), &mut targets, visited)?;
            }
        }
    }

    Ok(())
}

fn get_targets_with_hitlist(
    hitlist: &[String],
    targets: &mut HashSet<Target>,
) -> Result<(), io::Error> {
    let metadata = get_cargo_metadata(None)?;

    let mut workspace_hitlist: HashSet<&String> = HashSet::from_iter(hitlist);

    for package in metadata.packages {
        if workspace_hitlist.remove(&package.name) {
            for target in package.targets {
                targets.insert(Target::from_target(&target));
            }
        }
    }

    if workspace_hitlist.is_empty() {
        Ok(())
    } else {
        let package = workspace_hitlist.iter().next().unwrap();
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("package `{}` is not a member of the workspace", package),
        ))
    }
}

fn add_targets(target_paths: &[cargo_metadata::Target], targets: &mut HashSet<Target>) {
    for target in target_paths {
        targets.insert(Target::from_target(target));
    }
}

fn run_rustfmt(
    files: &[PathBuf],
    fmt_args: &[String],
    verbosity: Verbosity,
) -> Result<ExitStatus, io::Error> {
    let stdout = if verbosity == Verbosity::Quiet {
        std::process::Stdio::null()
    } else {
        std::process::Stdio::inherit()
    };

    if verbosity == Verbosity::Verbose {
        print!("rustfmt");
        for a in fmt_args {
            print!(" {}", a);
        }
        for f in files {
            print!(" {}", f.display());
        }
        println!();
    }

    let mut command = Command::new("rustfmt")
        .stdout(stdout)
        .args(files)
        .args(fmt_args)
        .spawn()
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => io::Error::new(
                io::ErrorKind::Other,
                "Could not run rustfmt, please make sure it is in your PATH.",
            ),
            _ => e,
        })?;

    command.wait()
}

fn get_cargo_metadata(manifest_path: Option<&Path>) -> Result<cargo_metadata::Metadata, io::Error> {
    match cargo_metadata::metadata(manifest_path) {
        Ok(metadata) => Ok(metadata),
        Err(..) => Err(io::Error::new(
            io::ErrorKind::Other,
            "`cargo manifest` failed.",
        )),
    }
}
