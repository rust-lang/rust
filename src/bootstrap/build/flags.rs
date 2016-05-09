// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Command-line interface of the rustbuild build system.
//!
//! This module implements the command-line parsing of the build system which
//! has various flags to configure how it's run.

use std::fs;
use std::path::PathBuf;
use std::process;
use std::slice;

use getopts::Options;

/// Deserialized version of all flags for this compile.
pub struct Flags {
    pub verbose: bool,
    pub stage: Option<u32>,
    pub build: String,
    pub host: Filter,
    pub target: Filter,
    pub step: Vec<String>,
    pub config: Option<PathBuf>,
    pub src: Option<PathBuf>,
    pub jobs: Option<u32>,
    pub args: Vec<String>,
    pub clean: bool,
}

pub struct Filter {
    values: Vec<String>,
}

impl Flags {
    pub fn parse(args: &[String]) -> Flags {
        let mut opts = Options::new();
        opts.optflag("v", "verbose", "use verbose output");
        opts.optopt("", "config", "TOML configuration file for build", "FILE");
        opts.optmulti("", "host", "host targets to build", "HOST");
        opts.reqopt("", "build", "build target of the stage0 compiler", "BUILD");
        opts.optmulti("", "target", "targets to build", "TARGET");
        opts.optmulti("s", "step", "build step to execute", "STEP");
        opts.optopt("", "stage", "stage to build", "N");
        opts.optopt("", "src", "path to repo root", "DIR");
        opts.optopt("j", "jobs", "number of jobs to run in parallel", "JOBS");
        opts.optflag("", "clean", "clean output directory");
        opts.optflag("h", "help", "print this help message");

        let usage = |n| -> ! {
            let brief = format!("Usage: rust.py [options]");
            print!("{}", opts.usage(&brief));
            process::exit(n);
        };

        let m = opts.parse(args).unwrap_or_else(|e| {
            println!("failed to parse options: {}", e);
            usage(1);
        });
        if m.opt_present("h") {
            usage(0);
        }

        let cfg_file = m.opt_str("config").map(PathBuf::from).or_else(|| {
            if fs::metadata("config.toml").is_ok() {
                Some(PathBuf::from("config.toml"))
            } else {
                None
            }
        });

        Flags {
            verbose: m.opt_present("v"),
            clean: m.opt_present("clean"),
            stage: m.opt_str("stage").map(|j| j.parse().unwrap()),
            build: m.opt_str("build").unwrap(),
            host: Filter { values: m.opt_strs("host") },
            target: Filter { values: m.opt_strs("target") },
            step: m.opt_strs("step"),
            config: cfg_file,
            src: m.opt_str("src").map(PathBuf::from),
            jobs: m.opt_str("jobs").map(|j| j.parse().unwrap()),
            args: m.free.clone(),
        }
    }
}

impl Filter {
    pub fn contains(&self, name: &str) -> bool {
        self.values.len() == 0 || self.values.iter().any(|s| s == name)
    }

    pub fn iter(&self) -> slice::Iter<String> {
        self.values.iter()
    }
}
