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
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};
use std::str;
use std::collections::HashSet;
use std::iter::FromIterator;

use json::Value;

use getopts::{Options, Matches};

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
    opts.optmulti("p",
                  "package",
                  "specify package to format (only usable in workspaces)",
                  "<package>");
    opts.optflag("", "all", "format all packages (only usable in workspaces)");

    let matches = match opts.parse(env::args().skip(1).take_while(|a| a != "--")) {
        Ok(m) => m,
        Err(e) => {
            print_usage(&opts, &e.to_string());
            return failure;
        }
    };

    let verbosity = match (matches.opt_present("v"), matches.opt_present("q")) {
        (false, false) => Verbosity::Normal,
        (false, true) => Verbosity::Quiet,
        (true, false) => Verbosity::Verbose,
        (true, true) => {
            print_usage(&opts, "quiet mode and verbose mode are not compatible");
            return failure;
        }
    };

    if matches.opt_present("h") {
        print_usage(&opts, "");
        return success;
    }

    let workspace_hitlist = WorkspaceHitlist::from_matches(&matches);

    match format_crate(verbosity, workspace_hitlist) {
        Err(e) => {
            print_usage(&opts, &e.to_string());
            failure
        }
        Ok(status) => {
            if status.success() {
                success
            } else {
                status.code().unwrap_or(failure)
            }
        }
    }
}

fn print_usage(opts: &Options, reason: &str) {
    let msg = format!("{}\nusage: cargo fmt [options]", reason);
    println!("{}\nThis utility formats all bin and lib files of the current crate using rustfmt. \
              Arguments after `--` are passed to rustfmt.",
             opts.usage(&msg));
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verbosity {
    Verbose,
    Normal,
    Quiet,
}

fn format_crate(verbosity: Verbosity,
                workspace_hitlist: WorkspaceHitlist)
                -> Result<ExitStatus, std::io::Error> {
    let targets = try!(get_targets(workspace_hitlist));

    // Currently only bin and lib files get formatted
    let files: Vec<_> = targets
        .into_iter()
        .filter(|t| t.kind.should_format())
        .inspect(|t| if verbosity == Verbosity::Verbose {
                     println!("[{:?}] {:?}", t.kind, t.path)
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
    Lib, // dylib, staticlib, lib
    Bin, // bin
    Example, // example file
    Test, // test file
    Bench, // bench file
    CustomBuild, // build script
    Other, // plugin,...
}

impl TargetKind {
    fn should_format(&self) -> bool {
        match *self {
            TargetKind::Lib | TargetKind::Bin | TargetKind::Example | TargetKind::Test |
            TargetKind::Bench | TargetKind::CustomBuild => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct Target {
    path: PathBuf,
    kind: TargetKind,
}

#[derive(Debug, PartialEq, Eq)]
pub enum WorkspaceHitlist {
    All,
    Some(Vec<String>),
    None,
}

impl WorkspaceHitlist {
    pub fn get_some<'a>(&'a self) -> Option<&'a [String]> {
        if let &WorkspaceHitlist::Some(ref hitlist) = self {
            Some(&hitlist)
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

// Returns a vector of all compile targets of a crate
fn get_targets(workspace_hitlist: WorkspaceHitlist) -> Result<Vec<Target>, std::io::Error> {
    let mut targets: Vec<Target> = vec![];
    if workspace_hitlist == WorkspaceHitlist::None {
        let output = try!(Command::new("cargo").arg("read-manifest").output());
        if output.status.success() {
            // None of the unwraps should fail if output of `cargo read-manifest` is correct
            let data = &String::from_utf8(output.stdout).unwrap();
            let json: Value = json::from_str(data).unwrap();
            let json_obj = json.as_object().unwrap();
            let jtargets = json_obj.get("targets").unwrap().as_array().unwrap();
            for jtarget in jtargets {
                targets.push(target_from_json(jtarget));
            }

            return Ok(targets);
        }
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                                       str::from_utf8(&output.stderr).unwrap()));
    }
    // This happens when cargo-fmt is not used inside a crate or
    // is used inside a workspace.
    // To ensure backward compatability, we only use `cargo metadata` for workspaces.
    // TODO: Is it possible only use metadata or read-manifest
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--no-deps")
        .output()?;
    if output.status.success() {
        let data = &String::from_utf8(output.stdout).unwrap();
        let json: Value = json::from_str(data).unwrap();
        let json_obj = json.as_object().unwrap();
        let mut hitlist: HashSet<&String> = if workspace_hitlist != WorkspaceHitlist::All {
            HashSet::from_iter(workspace_hitlist.get_some().unwrap())
        } else {
            HashSet::new() // Unused
        };
        let members: Vec<&Value> = json_obj
            .get("packages")
            .unwrap()
            .as_array()
            .unwrap()
            .into_iter()
            .filter(|member| if workspace_hitlist == WorkspaceHitlist::All {
                        true
                    } else {
                        let member_obj = member.as_object().unwrap();
                        let member_name = member_obj.get("name").unwrap().as_str().unwrap();
                        hitlist.take(&member_name.to_string()).is_some()
                    })
            .collect();
        if hitlist.len() != 0 {
            // Mimick cargo of only outputting one <package> spec.
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
                                           format!("package `{}` is not a member of the workspace",
                                                   hitlist.iter().next().unwrap())));
        }
        for member in members {
            let member_obj = member.as_object().unwrap();
            let jtargets = member_obj.get("targets").unwrap().as_array().unwrap();
            for jtarget in jtargets {
                targets.push(target_from_json(jtarget));
            }
        }
        return Ok(targets);
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                            str::from_utf8(&output.stderr).unwrap()))

}

fn target_from_json(jtarget: &Value) -> Target {
    let jtarget = jtarget.as_object().unwrap();
    let path = PathBuf::from(jtarget.get("src_path").unwrap().as_str().unwrap());
    let kinds = jtarget.get("kind").unwrap().as_array().unwrap();
    let kind = match kinds[0].as_str().unwrap() {
        "bin" => TargetKind::Bin,
        "lib" | "dylib" | "staticlib" | "cdylib" | "rlib" => TargetKind::Lib,
        "test" => TargetKind::Test,
        "example" => TargetKind::Example,
        "bench" => TargetKind::Bench,
        "custom-build" => TargetKind::CustomBuild,
        _ => TargetKind::Other,
    };

    Target {
        path: path,
        kind: kind,
    }
}

fn format_files(files: &[PathBuf],
                fmt_args: &[String],
                verbosity: Verbosity)
                -> Result<ExitStatus, std::io::Error> {
    let stdout = if verbosity == Verbosity::Quiet {
        std::process::Stdio::null()
    } else {
        std::process::Stdio::inherit()
    };
    if verbosity == Verbosity::Verbose {
        print!("rustfmt");
        for a in fmt_args.iter() {
            print!(" {}", a);
        }
        for f in files.iter() {
            print!(" {}", f.display());
        }
        println!("");
    }
    let mut command = try!(Command::new("rustfmt")
        .stdout(stdout)
        .args(files)
        .args(fmt_args)
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                std::io::Error::new(std::io::ErrorKind::Other,
                                    "Could not run rustfmt, please make sure it is in your PATH.")
            }
            _ => e,
        }));
    command.wait()
}
