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

extern crate getopts;
extern crate serde_json as json;

use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, ExitStatus};
use std::str;
use std::collections::HashSet;
use std::iter::FromIterator;

use getopts::{Matches, Options};
use json::Value;

fn main() {
    let exit_status = execute();
    std::io::stdout().flush().unwrap();
    std::process::exit(exit_status);
}

fn execute() -> i32 {
    let success = 0;
    let failure = 1;

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
    opts.optflag("", "all", "format all packages (only usable in workspaces)");

    // If there is any invalid argument passed to `cargo fmt`, return without formatting.
    if let Some(arg) = env::args()
        .skip(2)
        .take_while(|a| a != "--")
        .find(|a| !a.starts_with('-'))
    {
        print_usage_to_stderr(&opts, &format!("Invalid argument: `{}`.", arg));
        return failure;
    }

    let matches = match opts.parse(env::args().skip(1).take_while(|a| a != "--")) {
        Ok(m) => m,
        Err(e) => {
            print_usage_to_stderr(&opts, &e.to_string());
            return failure;
        }
    };

    let verbosity = match (matches.opt_present("v"), matches.opt_present("q")) {
        (false, false) => Verbosity::Normal,
        (false, true) => Verbosity::Quiet,
        (true, false) => Verbosity::Verbose,
        (true, true) => {
            print_usage_to_stderr(&opts, "quiet mode and verbose mode are not compatible");
            return failure;
        }
    };

    if matches.opt_present("h") {
        print_usage_to_stdout(&opts, "");
        return success;
    }

    let workspace_hitlist = WorkspaceHitlist::from_matches(&matches);

    match format_crate(verbosity, &workspace_hitlist) {
        Err(e) => {
            print_usage_to_stderr(&opts, &e.to_string());
            failure
        }
        Ok(status) => if status.success() {
            success
        } else {
            status.code().unwrap_or(failure)
        },
    }
}

macro_rules! print_usage {
    ($print:ident, $opts:ident, $reason:expr) => ({
        let msg = format!("{}\nusage: cargo fmt [options]", $reason);
        $print!(
            "{}\nThis utility formats all bin and lib files of the current crate using rustfmt. \
             Arguments after `--` are passed to rustfmt.",
            $opts.usage(&msg)
        );
    })
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

fn format_crate(
    verbosity: Verbosity,
    workspace_hitlist: &WorkspaceHitlist,
) -> Result<ExitStatus, io::Error> {
    let targets = get_targets(workspace_hitlist)?;

    // Currently only bin and lib files get formatted
    let files: Vec<_> = targets
        .into_iter()
        .filter(|t| t.kind.should_format())
        .inspect(|t| {
            if verbosity == Verbosity::Verbose {
                println!("[{:?}] {:?}", t.kind, t.path)
            }
        })
        .map(|t| t.path)
        .collect();

    format_files(&files, &get_fmt_args(), verbosity)
}

fn get_fmt_args() -> Vec<String> {
    // All arguments after -- are passed to rustfmt
    env::args().skip_while(|a| a != "--").skip(1).collect()
}

#[derive(Debug)]
enum TargetKind {
    Lib,         // dylib, staticlib, lib
    Bin,         // bin
    Example,     // example file
    Test,        // test file
    Bench,       // bench file
    CustomBuild, // build script
    ProcMacro,   // a proc macro implementation
    Other,       // plugin,...
}

impl TargetKind {
    fn should_format(&self) -> bool {
        match *self {
            TargetKind::Lib |
            TargetKind::Bin |
            TargetKind::Example |
            TargetKind::Test |
            TargetKind::Bench |
            TargetKind::CustomBuild |
            TargetKind::ProcMacro => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct Target {
    path: PathBuf,
    kind: TargetKind,
}

impl Target {
    pub fn from_json(json_val: &Value) -> Option<Self> {
        let jtarget = json_val.as_object()?;
        let path = PathBuf::from(jtarget.get("src_path")?.as_str()?);
        let kinds = jtarget.get("kind")?.as_array()?;
        let kind = match kinds[0].as_str()? {
            "bin" => TargetKind::Bin,
            "lib" | "dylib" | "staticlib" | "cdylib" | "rlib" => TargetKind::Lib,
            "test" => TargetKind::Test,
            "example" => TargetKind::Example,
            "bench" => TargetKind::Bench,
            "custom-build" => TargetKind::CustomBuild,
            "proc-macro" => TargetKind::ProcMacro,
            _ => TargetKind::Other,
        };

        Some(Target {
            path: path,
            kind: kind,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum WorkspaceHitlist {
    All,
    Some(Vec<String>),
    None,
}

impl WorkspaceHitlist {
    pub fn get_some(&self) -> Option<&[String]> {
        if let WorkspaceHitlist::Some(ref hitlist) = *self {
            Some(hitlist)
        } else {
            None
        }
    }

    pub fn from_matches(matches: &Matches) -> WorkspaceHitlist {
        match (matches.opt_present("all"), matches.opt_present("p")) {
            (false, false) => WorkspaceHitlist::None,
            (true, _) => WorkspaceHitlist::All,
            (false, true) => WorkspaceHitlist::Some(matches.opt_strs("p")),
        }
    }
}

fn get_cargo_metadata_from_utf8(v: &[u8]) -> Option<Value> {
    json::from_str(str::from_utf8(v).ok()?).ok()
}

fn get_json_array_with<'a>(v: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    v.as_object()?.get(key)?.as_array()
}

// `cargo metadata --no-deps | jq '.["packages"]'`
fn get_packages(v: &[u8]) -> Result<Vec<Value>, io::Error> {
    let e = io::Error::new(
        io::ErrorKind::NotFound,
        String::from("`cargo metadata` returned json without a 'packages' key"),
    );
    match get_cargo_metadata_from_utf8(v) {
        Some(ref json_obj) => get_json_array_with(json_obj, "packages").cloned().ok_or(e),
        None => Err(e),
    }
}

fn extract_target_from_package(package: &Value) -> Option<Vec<Target>> {
    let jtargets = get_json_array_with(package, "targets")?;
    let mut targets: Vec<Target> = vec![];
    for jtarget in jtargets {
        targets.push(Target::from_json(&jtarget)?);
    }
    Some(targets)
}

fn filter_packages_with_hitlist<'a>(
    packages: Vec<Value>,
    workspace_hitlist: &'a WorkspaceHitlist,
) -> Result<Vec<Value>, &'a String> {
    let some_hitlist: Option<HashSet<&String>> =
        workspace_hitlist.get_some().map(HashSet::from_iter);
    if some_hitlist.is_none() {
        return Ok(packages);
    }
    let mut hitlist = some_hitlist.unwrap();
    let members: Vec<Value> = packages
        .into_iter()
        .filter(|member| {
            member
                .as_object()
                .and_then(|member_obj| {
                    member_obj
                        .get("name")
                        .and_then(Value::as_str)
                        .map(|member_name| {
                            hitlist.take(&member_name.to_string()).is_some()
                        })
                })
                .unwrap_or(false)
        })
        .collect();
    if hitlist.is_empty() {
        Ok(members)
    } else {
        Err(hitlist.into_iter().next().unwrap())
    }
}

fn get_dependencies_from_package(package: &Value) -> Option<Vec<PathBuf>> {
    let jdependencies = get_json_array_with(package, "dependencies")?;
    let root_path = env::current_dir().ok()?;
    let mut dependencies: Vec<PathBuf> = vec![];
    for jdep in jdependencies {
        let jdependency = jdep.as_object()?;
        if !jdependency.get("source")?.is_null() {
            continue;
        }
        let name = jdependency.get("name")?.as_str()?;
        let mut path = root_path.clone();
        path.push(&name);
        dependencies.push(path);
    }
    Some(dependencies)
}

// Returns a vector of local dependencies under this crate
fn get_path_to_local_dependencies(packages: &[Value]) -> Vec<PathBuf> {
    let mut local_dependencies: Vec<PathBuf> = vec![];
    for package in packages {
        if let Some(mut d) = get_dependencies_from_package(package) {
            local_dependencies.append(&mut d);
        }
    }
    local_dependencies
}

// Returns a vector of all compile targets of a crate
fn get_targets(workspace_hitlist: &WorkspaceHitlist) -> Result<Vec<Target>, io::Error> {
    let output = Command::new("cargo")
        .args(&["metadata", "--no-deps", "--format-version=1"])
        .output()?;
    if output.status.success() {
        let cur_dir = env::current_dir()?;
        let mut targets: Vec<Target> = vec![];
        let packages = get_packages(&output.stdout)?;

        // If we can find any local dependencies, we will try to get targets from those as well.
        if *workspace_hitlist == WorkspaceHitlist::All {
            for path in get_path_to_local_dependencies(&packages) {
                match env::set_current_dir(path) {
                    Ok(..) => match get_targets(workspace_hitlist) {
                        Ok(ref mut t) => targets.append(t),
                        Err(..) => continue,
                    },
                    Err(..) => continue,
                }
            }
        }

        env::set_current_dir(cur_dir)?;
        match filter_packages_with_hitlist(packages, workspace_hitlist) {
            Ok(packages) => {
                for package in packages {
                    if let Some(mut target) = extract_target_from_package(&package) {
                        targets.append(&mut target);
                    }
                }
                Ok(targets)
            }
            Err(package) => {
                // Mimick cargo of only outputting one <package> spec.
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("package `{}` is not a member of the workspace", package),
                ))
            }
        }
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            str::from_utf8(&output.stderr).unwrap(),
        ))
    }
}

fn format_files(
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
        println!("");
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
