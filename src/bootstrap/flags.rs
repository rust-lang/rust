//! Command-line interface of the rustbuild build system.
//!
//! This module implements the command-line parsing of the build system which
//! has various flags to configure how it's run.

use std::fs;
use std::path::PathBuf;
use std::process;

use getopts::Options;

use crate::builder::Builder;
use crate::config::Config;
use crate::metadata;
use crate::{Build, DocTests};

use crate::cache::{Interned, INTERNER};

/// Deserialized version of all flags for this compile.
pub struct Flags {
    pub verbose: usize, // number of -v args; each extra -v after the first is passed to Cargo
    pub on_fail: Option<String>,
    pub stage: Option<u32>,
    pub keep_stage: Vec<u32>,

    pub host: Vec<Interned<String>>,
    pub target: Vec<Interned<String>>,
    pub config: Option<PathBuf>,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
    pub exclude: Vec<PathBuf>,
    pub rustc_error_format: Option<String>,
    pub dry_run: bool,

    // true => deny
    pub warnings: Option<bool>,
}

pub enum Subcommand {
    Build {
        paths: Vec<PathBuf>,
    },
    Check {
        paths: Vec<PathBuf>,
    },
    Clippy {
        paths: Vec<PathBuf>,
    },
    Fix {
        paths: Vec<PathBuf>,
    },
    Doc {
        paths: Vec<PathBuf>,
    },
    Test {
        paths: Vec<PathBuf>,
        /// Whether to automatically update stderr/stdout files
        bless: bool,
        compare_mode: Option<String>,
        pass: Option<String>,
        test_args: Vec<String>,
        rustc_args: Vec<String>,
        fail_fast: bool,
        doc_tests: DocTests,
        rustfix_coverage: bool,
    },
    Bench {
        paths: Vec<PathBuf>,
        test_args: Vec<String>,
    },
    Clean {
        all: bool,
    },
    Dist {
        paths: Vec<PathBuf>,
    },
    Install {
        paths: Vec<PathBuf>,
    },
}

impl Default for Subcommand {
    fn default() -> Subcommand {
        Subcommand::Build {
            paths: vec![PathBuf::from("nowhere")],
        }
    }
}

impl Flags {
    pub fn parse(args: &[String]) -> Flags {
        let mut extra_help = String::new();
        let mut subcommand_help = String::from("\
Usage: x.py <subcommand> [options] [<paths>...]

Subcommands:
    build       Compile either the compiler or libraries
    check       Compile either the compiler or libraries, using cargo check
    clippy      Run clippy
    fix         Run cargo fix
    test        Build and run some test suites
    bench       Build and run some benchmarks
    doc         Build documentation
    clean       Clean out build directories
    dist        Build distribution artifacts
    install     Install distribution artifacts

To learn more about a subcommand, run `./x.py <subcommand> -h`"
        );

        let mut opts = Options::new();
        // Options common to all subcommands
        opts.optflagmulti("v", "verbose", "use verbose output (-vv for very verbose)");
        opts.optflag("i", "incremental", "use incremental compilation");
        opts.optopt("", "config", "TOML configuration file for build", "FILE");
        opts.optopt("", "build", "build target of the stage0 compiler", "BUILD");
        opts.optmulti("", "host", "host targets to build", "HOST");
        opts.optmulti("", "target", "target targets to build", "TARGET");
        opts.optmulti("", "exclude", "build paths to exclude", "PATH");
        opts.optopt("", "on-fail", "command to run on failure", "CMD");
        opts.optflag("", "dry-run", "dry run; don't build anything");
        opts.optopt("", "stage",
            "stage to build (indicates compiler to use/test, e.g., stage 0 uses the \
             bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)",
            "N");
        opts.optmulti("", "keep-stage", "stage(s) to keep without recompiling \
            (pass multiple times to keep e.g., both stages 0 and 1)", "N");
        opts.optopt("", "src", "path to the root of the rust checkout", "DIR");
        opts.optopt("j", "jobs", "number of jobs to run in parallel", "JOBS");
        opts.optflag("h", "help", "print this help message");
        opts.optopt(
            "",
            "warnings",
            "if value is deny, will deny warnings, otherwise use default",
            "VALUE",
        );
        opts.optopt("", "error-format", "rustc error format", "FORMAT");

        // fn usage()
        let usage =
            |exit_code: i32, opts: &Options, subcommand_help: &str, extra_help: &str| -> ! {
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
        let subcommand = args.iter().find(|&s| {
            (s == "build")
                || (s == "check")
                || (s == "clippy")
                || (s == "fix")
                || (s == "test")
                || (s == "bench")
                || (s == "doc")
                || (s == "clean")
                || (s == "dist")
                || (s == "install")
        });
        let subcommand = match subcommand {
            Some(s) => s,
            None => {
                // No or an invalid subcommand -- show the general usage and subcommand help
                // An exit code will be 0 when no subcommand is given, and 1 in case of an invalid
                // subcommand.
                println!("{}\n", subcommand_help);
                let exit_code = if args.is_empty() { 0 } else { 1 };
                process::exit(exit_code);
            }
        };

        // Some subcommands get extra options
        match subcommand.as_str() {
            "test" => {
                opts.optflag("", "no-fail-fast", "Run all tests regardless of failure");
                opts.optmulti("", "test-args", "extra arguments", "ARGS");
                opts.optmulti(
                    "",
                    "rustc-args",
                    "extra options to pass the compiler when running tests",
                    "ARGS",
                );
                opts.optflag("", "no-doc", "do not run doc tests");
                opts.optflag("", "doc", "only run doc tests");
                opts.optflag(
                    "",
                    "bless",
                    "update all stderr/stdout files of failing ui tests",
                );
                opts.optopt(
                    "",
                    "compare-mode",
                    "mode describing what file the actual ui output will be compared to",
                    "COMPARE MODE",
                );
                opts.optopt(
                    "",
                    "pass",
                    "force {check,build,run}-pass tests to this mode.",
                    "check | build | run"
                );
                opts.optflag(
                    "",
                    "rustfix-coverage",
                    "enable this to generate a Rustfix coverage file, which is saved in \
                        `/<build_base>/rustfix_missing_coverage.txt`",
                );
            }
            "bench" => {
                opts.optmulti("", "test-args", "extra arguments", "ARGS");
            }
            "clean" => {
                opts.optflag("", "all", "clean all build artifacts");
            }
            _ => {}
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
                if check_subcommand != subcommand {
                    pass_sanity_check = false;
                }
            }
            None => {
                pass_sanity_check = false;
            }
        }
        if !pass_sanity_check {
            println!("{}\n", subcommand_help);
            println!(
                "Sorry, I couldn't figure out which subcommand you were trying to specify.\n\
                 You may need to move some options to after the subcommand.\n"
            );
            process::exit(1);
        }
        // Extra help text for some commands
        match subcommand.as_str() {
            "build" => {
                subcommand_help.push_str(
                    "\n
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

    For a quick build of a usable compiler, you can pass:

        ./x.py build --stage 1 src/libtest

    This will first build everything once (like `--stage 0` without further
    arguments would), and then use the compiler built in stage 0 to build
    src/libtest and its dependencies.
    Once this is done, build/$ARCH/stage1 contains a usable compiler.",
                );
            }
            "check" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to compile. For example:

        ./x.py check src/libcore
        ./x.py check src/libcore src/libproc_macro

    If no arguments are passed then the complete artifacts are compiled: std, test, and rustc. Note
    also that since we use `cargo check`, by default this will automatically enable incremental
    compilation, so there's no need to pass it separately, though it won't hurt. We also completely
    ignore the stage passed, as there's no way to compile in non-stage 0 without actually building
    the compiler.",
                );
            }
            "clippy" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to run clippy against. For example:

        ./x.py clippy src/libcore
        ./x.py clippy src/libcore src/libproc_macro",
                );
            }
            "fix" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to run `cargo fix` against. For example:

        ./x.py fix src/libcore
        ./x.py fix src/libcore src/libproc_macro",
                );
            }
            "test" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to tests that
    should be compiled and run. For example:

        ./x.py test src/test/run-pass
        ./x.py test src/libstd --test-args hash_map
        ./x.py test src/libstd --stage 0 --no-doc
        ./x.py test src/test/ui --bless
        ./x.py test src/test/ui --compare-mode nll

    Note that `test src/test/* --stage N` does NOT depend on `build src/rustc --stage N`;
    just like `build src/libstd --stage N` it tests the compiler produced by the previous
    stage.

    If no arguments are passed then the complete artifacts for that stage are
    compiled and tested.

        ./x.py test
        ./x.py test --stage 1",
                );
            }
            "doc" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories of documentation
    to build. For example:

        ./x.py doc src/doc/book
        ./x.py doc src/doc/nomicon
        ./x.py doc src/doc/book src/libstd

    If no arguments are passed then everything is documented:

        ./x.py doc
        ./x.py doc --stage 1",
                );
            }
            _ => {}
        };
        // Get any optional paths which occur after the subcommand
        let paths = matches.free[1..]
            .iter()
            .map(|p| p.into())
            .collect::<Vec<PathBuf>>();

        let cfg_file = matches.opt_str("config").map(PathBuf::from).or_else(|| {
            if fs::metadata("config.toml").is_ok() {
                Some(PathBuf::from("config.toml"))
            } else {
                None
            }
        });

        // All subcommands except `clean` can have an optional "Available paths" section
        if matches.opt_present("verbose") {
            let config = Config::parse(&["build".to_string()]);
            let mut build = Build::new(config);
            metadata::build(&mut build);

            let maybe_rules_help = Builder::get_help(&build, subcommand.as_str());
            extra_help.push_str(maybe_rules_help.unwrap_or_default().as_str());
        } else if subcommand.as_str() != "clean" {
            extra_help.push_str(
                format!(
                    "Run `./x.py {} -h -v` to see a list of available paths.",
                    subcommand
                ).as_str(),
            );
        }

        // User passed in -h/--help?
        if matches.opt_present("help") {
            usage(0, &opts, &subcommand_help, &extra_help);
        }

        let cmd = match subcommand.as_str() {
            "build" => Subcommand::Build { paths },
            "check" => Subcommand::Check { paths },
            "clippy" => Subcommand::Clippy { paths },
            "fix" => Subcommand::Fix { paths },
            "test" => Subcommand::Test {
                paths,
                bless: matches.opt_present("bless"),
                compare_mode: matches.opt_str("compare-mode"),
                pass: matches.opt_str("pass"),
                test_args: matches.opt_strs("test-args"),
                rustc_args: matches.opt_strs("rustc-args"),
                fail_fast: !matches.opt_present("no-fail-fast"),
                rustfix_coverage: matches.opt_present("rustfix-coverage"),
                doc_tests: if matches.opt_present("doc") {
                    DocTests::Only
                } else if matches.opt_present("no-doc") {
                    DocTests::No
                } else {
                    DocTests::Yes
                },
            },
            "bench" => Subcommand::Bench {
                paths,
                test_args: matches.opt_strs("test-args"),
            },
            "doc" => Subcommand::Doc { paths },
            "clean" => {
                if !paths.is_empty() {
                    println!("\nclean does not take a path argument\n");
                    usage(1, &opts, &subcommand_help, &extra_help);
                }

                Subcommand::Clean {
                    all: matches.opt_present("all"),
                }
            }
            "dist" => Subcommand::Dist { paths },
            "install" => Subcommand::Install { paths },
            _ => {
                usage(1, &opts, &subcommand_help, &extra_help);
            }
        };

        Flags {
            verbose: matches.opt_count("verbose"),
            stage: matches.opt_str("stage").map(|j| j.parse().unwrap()),
            dry_run: matches.opt_present("dry-run"),
            on_fail: matches.opt_str("on-fail"),
            rustc_error_format: matches.opt_str("error-format"),
            keep_stage: matches.opt_strs("keep-stage")
                .into_iter().map(|j| j.parse().unwrap())
                .collect(),
            host: split(&matches.opt_strs("host"))
                .into_iter()
                .map(|x| INTERNER.intern_string(x))
                .collect::<Vec<_>>(),
            target: split(&matches.opt_strs("target"))
                .into_iter()
                .map(|x| INTERNER.intern_string(x))
                .collect::<Vec<_>>(),
            config: cfg_file,
            jobs: matches.opt_str("jobs").map(|j| j.parse().unwrap()),
            cmd,
            incremental: matches.opt_present("incremental"),
            exclude: split(&matches.opt_strs("exclude"))
                .into_iter()
                .map(|p| p.into())
                .collect::<Vec<_>>(),
            warnings: matches.opt_str("warnings").map(|v| v == "deny"),
        }
    }
}

impl Subcommand {
    pub fn test_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref test_args, .. } | Subcommand::Bench { ref test_args, .. } => {
                test_args
                    .iter()
                    .flat_map(|s| s.split_whitespace())
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn rustc_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref rustc_args, .. } => rustc_args
                .iter()
                .flat_map(|s| s.split_whitespace())
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn fail_fast(&self) -> bool {
        match *self {
            Subcommand::Test { fail_fast, .. } => fail_fast,
            _ => false,
        }
    }

    pub fn doc_tests(&self) -> DocTests {
        match *self {
            Subcommand::Test { doc_tests, .. } => doc_tests,
            _ => DocTests::Yes,
        }
    }

    pub fn bless(&self) -> bool {
        match *self {
            Subcommand::Test { bless, .. } => bless,
            _ => false,
        }
    }

    pub fn rustfix_coverage(&self) -> bool {
        match *self {
            Subcommand::Test { rustfix_coverage, .. } => rustfix_coverage,
            _ => false,
        }
    }

    pub fn compare_mode(&self) -> Option<&str> {
        match *self {
            Subcommand::Test {
                ref compare_mode, ..
            } => compare_mode.as_ref().map(|s| &s[..]),
            _ => None,
        }
    }

    pub fn pass(&self) -> Option<&str> {
        match *self {
            Subcommand::Test {
                ref pass, ..
            } => pass.as_ref().map(|s| &s[..]),
            _ => None,
        }
    }
}

fn split(s: &[String]) -> Vec<String> {
    s.iter()
        .flat_map(|s| s.split(','))
        .map(|s| s.to_string())
        .collect()
}
