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

use getopts::Options;

use Build;
use config::Config;
use metadata;
use step;

/// Deserialized version of all flags for this compile.
pub struct Flags {
    pub verbose: usize, // verbosity level: 0 == not verbose, 1 == verbose, 2 == very verbose
    pub on_fail: Option<String>,
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
        no_fail_fast: bool,
    },
    Bench {
        paths: Vec<PathBuf>,
        test_args: Vec<String>,
    },
    Clean,
    Dist {
        paths: Vec<PathBuf>,
    },
    Install {
        paths: Vec<PathBuf>,
    },
}

impl Flags {
    pub fn parse(args: &[String]) -> Flags {
        let mut extra_help = String::new();
        let mut subcommand_help = format!("\
Usage: x.py <subcommand> [options] [<paths>...]

Subcommands:
    build       Compile either the compiler or libraries
    test        Build and run some test suites
    bench       Build and run some benchmarks
    doc         Build documentation
    clean       Clean out build directories
    dist        Build distribution artifacts
    install     Install distribution artifacts

To learn more about a subcommand, run `./x.py <subcommand> -h`");

        let mut opts = Options::new();
        // Options common to all subcommands
        opts.optflagmulti("v", "verbose", "use verbose output (-vv for very verbose)");
        opts.optflag("i", "incremental", "use incremental compilation");
        opts.optopt("", "config", "TOML configuration file for build", "FILE");
        opts.optopt("", "build", "build target of the stage0 compiler", "BUILD");
        opts.optmulti("", "host", "host targets to build", "HOST");
        opts.optmulti("", "target", "target targets to build", "TARGET");
        opts.optopt("", "on-fail", "command to run on failure", "CMD");
        opts.optopt("", "stage", "stage to build", "N");
        opts.optopt("", "keep-stage", "stage to keep without recompiling", "N");
        opts.optopt("", "src", "path to the root of the rust checkout", "DIR");
        opts.optopt("j", "jobs", "number of jobs to run in parallel", "JOBS");
        opts.optflag("h", "help", "print this help message");

        // fn usage()
        let usage = |exit_code: i32, opts: &Options, subcommand_help: &str, extra_help: &str| -> ! {
            println!("{}", opts.usage(subcommand_help));
            if !extra_help.is_empty() {
                println!("{}", extra_help);
            }
            process::exit(exit_code);
        };

        // We can't use getopt to parse the options until we have completed specifying which
        // options are valid, but under the current implementation, some options are conditional on
        // the subcommand. Therefore we must manually identify the subcommand first, so that we can
        // complete the definition of the options.  Then we can use the getopt::Matches object from
        // there on out.
        let mut possible_subcommands = args.iter().collect::<Vec<_>>();
        possible_subcommands.retain(|&s|
                                           (s == "build")
                                        || (s == "test")
                                        || (s == "bench")
                                        || (s == "doc")
                                        || (s == "clean")
                                        || (s == "dist")
                                        || (s == "install"));
        let subcommand = match possible_subcommands.first() {
            Some(s) => s,
            None => {
                // No subcommand -- show the general usage and subcommand help
                println!("{}\n", subcommand_help);
                process::exit(0);
            }
        };

        // Some subcommands get extra options
        match subcommand.as_str() {
            "test"  => {
                opts.optflag("", "no-fail-fast", "Run all tests regardless of failure");
                opts.optmulti("", "test-args", "extra arguments", "ARGS");
            },
            "bench" => { opts.optmulti("", "test-args", "extra arguments", "ARGS"); },
            _ => { },
        };

        // Done specifying what options are possible, so do the getopts parsing
        let matches = opts.parse(&args[..]).unwrap_or_else(|e| {
            // Invalid argument/option format
            println!("\n{}\n", e);
            usage(1, &opts, &subcommand_help, &extra_help);
        });
        // Extra sanity check to make sure we didn't hit this crazy corner case:
        //
        //     ./x.py --frobulate clean build
        //            ^-- option  ^     ^- actual subcommand
        //                        \_ arg to option could be mistaken as subcommand
        let mut pass_sanity_check = true;
        match matches.free.get(0) {
            Some(check_subcommand) => {
                if &check_subcommand != subcommand {
                    pass_sanity_check = false;
                }
            },
            None => {
                pass_sanity_check = false;
            }
        }
        if !pass_sanity_check {
            println!("{}\n", subcommand_help);
            println!("Sorry, I couldn't figure out which subcommand you were trying to specify.\n\
                      You may need to move some options to after the subcommand.\n");
            process::exit(1);
        }
        // Extra help text for some commands
        match subcommand.as_str() {
            "build" => {
                subcommand_help.push_str("\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to compile. For example:

        ./x.py build src/libcore
        ./x.py build src/libcore src/libproc_macro
        ./x.py build src/libstd --stage 1

    If no arguments are passed then the complete artifacts for that stage are
    also compiled.

        ./x.py build
        ./x.py build --stage 1

    For a quick build with a usable compile, you can pass:

        ./x.py build --stage 1 src/libtest

    This will first build everything once (like --stage 0 without further
    arguments would), and then use the compiler built in stage 0 to build
    src/libtest and its dependencies.");
            }
            "test" => {
                subcommand_help.push_str("\n
Arguments:
    This subcommand accepts a number of paths to directories to tests that
    should be compiled and run. For example:

        ./x.py test src/test/run-pass
        ./x.py test src/libstd --test-args hash_map
        ./x.py test src/libstd --stage 0

    If no arguments are passed then the complete artifacts for that stage are
    compiled and tested.

        ./x.py test
        ./x.py test --stage 1");
            }
            "doc" => {
                subcommand_help.push_str("\n
Arguments:
    This subcommand accepts a number of paths to directories of documentation
    to build. For example:

        ./x.py doc src/doc/book
        ./x.py doc src/doc/nomicon
        ./x.py doc src/doc/book src/libstd

    If no arguments are passed then everything is documented:

        ./x.py doc
        ./x.py doc --stage 1");
            }
            _ => { }
        };
        // Get any optional paths which occur after the subcommand
        let cwd = t!(env::current_dir());
        let paths = matches.free[1..].iter().map(|p| cwd.join(p)).collect::<Vec<_>>();


        // All subcommands can have an optional "Available paths" section
        if matches.opt_present("verbose") {
            let flags = Flags::parse(&["build".to_string()]);
            let mut config = Config::default();
            config.build = flags.build.clone();
            let mut build = Build::new(flags, config);
            metadata::build(&mut build);
            let maybe_rules_help = step::build_rules(&build).get_help(subcommand);
            if maybe_rules_help.is_some() {
                extra_help.push_str(maybe_rules_help.unwrap().as_str());
            }
        } else {
            extra_help.push_str(format!("Run `./x.py {} -h -v` to see a list of available paths.",
                     subcommand).as_str());
        }

        // User passed in -h/--help?
        if matches.opt_present("help") {
            usage(0, &opts, &subcommand_help, &extra_help);
        }

        let cmd = match subcommand.as_str() {
            "build" => {
                Subcommand::Build { paths: paths }
            }
            "test" => {
                Subcommand::Test {
                    paths: paths,
                    test_args: matches.opt_strs("test-args"),
                    no_fail_fast: matches.opt_present("no-fail-fast"),
                }
            }
            "bench" => {
                Subcommand::Bench {
                    paths: paths,
                    test_args: matches.opt_strs("test-args"),
                }
            }
            "doc" => {
                Subcommand::Doc { paths: paths }
            }
            "clean" => {
                if paths.len() > 0 {
                    println!("\nclean takes no arguments\n");
                    usage(1, &opts, &subcommand_help, &extra_help);
                }
                Subcommand::Clean
            }
            "dist" => {
                Subcommand::Dist {
                    paths: paths,
                }
            }
            "install" => {
                Subcommand::Install {
                    paths: paths,
                }
            }
            _ => {
                usage(1, &opts, &subcommand_help, &extra_help);
            }
        };


        let cfg_file = matches.opt_str("config").map(PathBuf::from).or_else(|| {
            if fs::metadata("config.toml").is_ok() {
                Some(PathBuf::from("config.toml"))
            } else {
                None
            }
        });

        let mut stage = matches.opt_str("stage").map(|j| j.parse().unwrap());

        if matches.opt_present("incremental") {
            if stage.is_none() {
                stage = Some(1);
            }
        }

        Flags {
            verbose: matches.opt_count("verbose"),
            stage: stage,
            on_fail: matches.opt_str("on-fail"),
            keep_stage: matches.opt_str("keep-stage").map(|j| j.parse().unwrap()),
            build: matches.opt_str("build").unwrap_or_else(|| {
                env::var("BUILD").unwrap()
            }),
            host: split(matches.opt_strs("host")),
            target: split(matches.opt_strs("target")),
            config: cfg_file,
            src: matches.opt_str("src").map(PathBuf::from),
            jobs: matches.opt_str("jobs").map(|j| j.parse().unwrap()),
            cmd: cmd,
            incremental: matches.opt_present("incremental"),
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

    pub fn no_fail_fast(&self) -> bool {
        match *self {
            Subcommand::Test { no_fail_fast, .. } => no_fail_fast,
            _ => false,
        }
    }
}

fn split(s: Vec<String>) -> Vec<String> {
    s.iter().flat_map(|s| s.split(',')).map(|s| s.to_string()).collect()
}
