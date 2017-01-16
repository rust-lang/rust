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

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use getopts::{Matches, Options};

use Build;
use config::Config;
use metadata;
use step;

/// Deserialized version of all flags for this compile.
pub struct Flags {
    pub verbose: usize, // verbosity level: 0 == not verbose, 1 == verbose, 2 == very verbose
    pub stage: Option<u32>,
    pub keep_stage: Option<u32>,
    pub build: String,
    pub host: Vec<String>,
    pub target: Vec<String>,
    pub config: Option<PathBuf>,
    pub src: Option<PathBuf>,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
}

impl Flags {
    pub fn verbose(&self) -> bool {
        self.verbose > 0
    }

    pub fn very_verbose(&self) -> bool {
        self.verbose > 1
    }
}

pub enum Subcommand {
    Build {
        paths: Vec<PathBuf>,
    },
    Doc {
        paths: Vec<PathBuf>,
    },
    Test {
        paths: Vec<PathBuf>,
        test_args: Vec<String>,
    },
    Bench {
        paths: Vec<PathBuf>,
        test_args: Vec<String>,
    },
    Clean,
    Dist {
        paths: Vec<PathBuf>,
        install: bool,
    },
}

impl Flags {
    pub fn parse(args: &[String]) -> Flags {
        let mut opts = Options::new();
        opts.optflagmulti("v", "verbose", "use verbose output (-vv for very verbose)");
        opts.optflag("i", "incremental", "use incremental compilation");
        opts.optopt("", "config", "TOML configuration file for build", "FILE");
        opts.optopt("", "build", "build target of the stage0 compiler", "BUILD");
        opts.optmulti("", "host", "host targets to build", "HOST");
        opts.optmulti("", "target", "target targets to build", "TARGET");
        opts.optopt("", "stage", "stage to build", "N");
        opts.optopt("", "keep-stage", "stage to keep without recompiling", "N");
        opts.optopt("", "src", "path to the root of the rust checkout", "DIR");
        opts.optopt("j", "jobs", "number of jobs to run in parallel", "JOBS");
        opts.optflag("h", "help", "print this help message");

        let usage = |n, opts: &Options| -> ! {
            let command = args.get(0).map(|s| &**s);
            let brief = format!("Usage: x.py {} [options] [<args>...]",
                                command.unwrap_or("<command>"));

            println!("{}", opts.usage(&brief));
            match command {
                Some("build") => {
                    println!("\
Arguments:
    This subcommand accepts a number of positional arguments of directories to
    the crates and/or artifacts to compile. For example:

        ./x.py build src/libcore
        ./x.py build src/libproc_macro
        ./x.py build src/libstd --stage 1

    If no arguments are passed then the complete artifacts for that stage are
    also compiled.

        ./x.py build
        ./x.py build --stage 1

    For a quick build with a usable compile, you can pass:

        ./x.py build --stage 1 src/libtest
");
                }

                Some("test") => {
                    println!("\
Arguments:
    This subcommand accepts a number of positional arguments of directories to
    tests that should be compiled and run. For example:

        ./x.py test src/test/run-pass
        ./x.py test src/libstd --test-args hash_map
        ./x.py test src/libstd --stage 0

    If no arguments are passed then the complete artifacts for that stage are
    compiled and tested.

        ./x.py test
        ./x.py test --stage 1
");
                }

                Some("doc") => {
                    println!("\
Arguments:
    This subcommand accepts a number of positional arguments of directories of
    documentation to build. For example:

        ./x.py doc src/doc/book
        ./x.py doc src/doc/nomicon
        ./x.py doc src/libstd

    If no arguments are passed then everything is documented:

        ./x.py doc
        ./x.py doc --stage 1
");
                }

                _ => {}
            }

            if let Some(command) = command {
                if command == "build" ||
                   command == "dist" ||
                   command == "doc" ||
                   command == "test" ||
                   command == "bench" ||
                   command == "clean"  {
                    println!("Available invocations:");
                    if args.iter().any(|a| a == "-v") {
                        let flags = Flags::parse(&["build".to_string()]);
                        let mut config = Config::default();
                        config.build = flags.build.clone();
                        let mut build = Build::new(flags, config);
                        metadata::build(&mut build);
                        step::build_rules(&build).print_help(command);
                    } else {
                        println!("    ... elided, run `./x.py {} -h -v` to see",
                                 command);
                    }

                    println!("");
                }
            }

println!("\
Subcommands:
    build       Compile either the compiler or libraries
    test        Build and run some test suites
    bench       Build and run some benchmarks
    doc         Build documentation
    clean       Clean out build directories
    dist        Build and/or install distribution artifacts

To learn more about a subcommand, run `./x.py <command> -h`
");

            process::exit(n);
        };
        if args.len() == 0 {
            println!("a command must be passed");
            usage(1, &opts);
        }
        let parse = |opts: &Options| {
            let m = opts.parse(&args[1..]).unwrap_or_else(|e| {
                println!("failed to parse options: {}", e);
                usage(1, opts);
            });
            if m.opt_present("h") {
                usage(0, opts);
            }
            return m
        };

        let cwd = t!(env::current_dir());
        let remaining_as_path = |m: &Matches| {
            m.free.iter().map(|p| cwd.join(p)).collect::<Vec<_>>()
        };

        let m: Matches;
        let cmd = match &args[0][..] {
            "build" => {
                m = parse(&opts);
                Subcommand::Build { paths: remaining_as_path(&m) }
            }
            "doc" => {
                m = parse(&opts);
                Subcommand::Doc { paths: remaining_as_path(&m) }
            }
            "test" => {
                opts.optmulti("", "test-args", "extra arguments", "ARGS");
                m = parse(&opts);
                Subcommand::Test {
                    paths: remaining_as_path(&m),
                    test_args: m.opt_strs("test-args"),
                }
            }
            "bench" => {
                opts.optmulti("", "test-args", "extra arguments", "ARGS");
                m = parse(&opts);
                Subcommand::Bench {
                    paths: remaining_as_path(&m),
                    test_args: m.opt_strs("test-args"),
                }
            }
            "clean" => {
                m = parse(&opts);
                if m.free.len() > 0 {
                    println!("clean takes no arguments");
                    usage(1, &opts);
                }
                Subcommand::Clean
            }
            "dist" => {
                opts.optflag("", "install", "run installer as well");
                m = parse(&opts);
                Subcommand::Dist {
                    paths: remaining_as_path(&m),
                    install: m.opt_present("install"),
                }
            }
            "--help" => usage(0, &opts),
            cmd => {
                println!("unknown command: {}", cmd);
                usage(1, &opts);
            }
        };


        let cfg_file = m.opt_str("config").map(PathBuf::from).or_else(|| {
            if fs::metadata("config.toml").is_ok() {
                Some(PathBuf::from("config.toml"))
            } else {
                None
            }
        });

        let mut stage = m.opt_str("stage").map(|j| j.parse().unwrap());

        let incremental = m.opt_present("i");

        if incremental {
            if stage.is_none() {
                stage = Some(1);
            }
        }

        Flags {
            verbose: m.opt_count("v"),
            stage: stage,
            keep_stage: m.opt_str("keep-stage").map(|j| j.parse().unwrap()),
            build: m.opt_str("build").unwrap_or_else(|| {
                env::var("BUILD").unwrap()
            }),
            host: split(m.opt_strs("host")),
            target: split(m.opt_strs("target")),
            config: cfg_file,
            src: m.opt_str("src").map(PathBuf::from),
            jobs: m.opt_str("jobs").map(|j| j.parse().unwrap()),
            cmd: cmd,
            incremental: incremental,
        }
    }
}

impl Subcommand {
    pub fn test_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref test_args, .. } |
            Subcommand::Bench { ref test_args, .. } => {
                test_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => Vec::new(),
        }
    }
}

fn split(s: Vec<String>) -> Vec<String> {
    s.iter().flat_map(|s| s.split(',')).map(|s| s.to_string()).collect()
}
