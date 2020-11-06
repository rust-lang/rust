//! Command-line interface of the rustbuild build system.
//!
//! This module implements the command-line parsing of the build system which
//! has various flags to configure how it's run.

use std::env;
use std::path::PathBuf;
use std::process;

use build_helper::t;
use getopts::Options;

use crate::builder::Builder;
use crate::config::{Config, TargetSelection};
use crate::setup::Profile;
use crate::{Build, DocTests};

/// Deserialized version of all flags for this compile.
pub struct Flags {
    pub verbose: usize, // number of -v args; each extra -v after the first is passed to Cargo
    pub on_fail: Option<String>,
    pub stage: Option<u32>,
    pub keep_stage: Vec<u32>,
    pub keep_stage_std: Vec<u32>,

    pub host: Option<Vec<TargetSelection>>,
    pub target: Option<Vec<TargetSelection>>,
    pub config: Option<PathBuf>,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
    pub exclude: Vec<PathBuf>,
    pub include_default_paths: bool,
    pub rustc_error_format: Option<String>,
    pub json_output: bool,
    pub dry_run: bool,

    // This overrides the deny-warnings configuration option,
    // which passes -Dwarnings to the compiler invocations.
    //
    // true => deny, false => warn
    pub deny_warnings: Option<bool>,

    pub llvm_skip_rebuild: Option<bool>,
}

pub enum Subcommand {
    Build {
        paths: Vec<PathBuf>,
    },
    Check {
        // Whether to run checking over all targets (e.g., unit / integration
        // tests).
        all_targets: bool,
        paths: Vec<PathBuf>,
    },
    Clippy {
        fix: bool,
        paths: Vec<PathBuf>,
    },
    Fix {
        paths: Vec<PathBuf>,
    },
    Format {
        check: bool,
    },
    Doc {
        paths: Vec<PathBuf>,
        open: bool,
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
    Run {
        paths: Vec<PathBuf>,
    },
    Setup {
        profile: Profile,
    },
}

impl Default for Subcommand {
    fn default() -> Subcommand {
        Subcommand::Build { paths: vec![PathBuf::from("nowhere")] }
    }
}

impl Flags {
    pub fn parse(args: &[String]) -> Flags {
        let mut subcommand_help = String::from(
            "\
Usage: x.py <subcommand> [options] [<paths>...]

Subcommands:
    build, b    Compile either the compiler or libraries
    check, c    Compile either the compiler or libraries, using cargo check
    clippy      Run clippy (uses rustup/cargo-installed clippy binary)
    fix         Run cargo fix
    fmt         Run rustfmt
    test, t     Build and run some test suites
    bench       Build and run some benchmarks
    doc         Build documentation
    clean       Clean out build directories
    dist        Build distribution artifacts
    install     Install distribution artifacts
    run, r      Run tools contained in this repository
    setup       Create a config.toml (making it easier to use `x.py` itself)

To learn more about a subcommand, run `./x.py <subcommand> -h`",
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
        opts.optflag(
            "",
            "include-default-paths",
            "include default paths in addition to the provided ones",
        );
        opts.optopt("", "on-fail", "command to run on failure", "CMD");
        opts.optflag("", "dry-run", "dry run; don't build anything");
        opts.optopt(
            "",
            "stage",
            "stage to build (indicates compiler to use/test, e.g., stage 0 uses the \
             bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)",
            "N",
        );
        opts.optmulti(
            "",
            "keep-stage",
            "stage(s) to keep without recompiling \
            (pass multiple times to keep e.g., both stages 0 and 1)",
            "N",
        );
        opts.optmulti(
            "",
            "keep-stage-std",
            "stage(s) of the standard library to keep without recompiling \
            (pass multiple times to keep e.g., both stages 0 and 1)",
            "N",
        );
        opts.optopt("", "src", "path to the root of the rust checkout", "DIR");
        let j_msg = format!(
            "number of jobs to run in parallel; \
             defaults to {} (this host's logical CPU count)",
            num_cpus::get()
        );
        opts.optopt("j", "jobs", &j_msg, "JOBS");
        opts.optflag("h", "help", "print this help message");
        opts.optopt(
            "",
            "warnings",
            "if value is deny, will deny warnings, otherwise use default",
            "VALUE",
        );
        opts.optopt("", "error-format", "rustc error format", "FORMAT");
        opts.optflag("", "json-output", "use message-format=json");
        opts.optopt(
            "",
            "llvm-skip-rebuild",
            "whether rebuilding llvm should be skipped \
             a VALUE of TRUE indicates that llvm will not be rebuilt \
             VALUE overrides the skip-rebuild option in config.toml.",
            "VALUE",
        );

        // We can't use getopt to parse the options until we have completed specifying which
        // options are valid, but under the current implementation, some options are conditional on
        // the subcommand. Therefore we must manually identify the subcommand first, so that we can
        // complete the definition of the options.  Then we can use the getopt::Matches object from
        // there on out.
        let subcommand = args.iter().find(|&s| {
            (s == "build")
                || (s == "b")
                || (s == "check")
                || (s == "c")
                || (s == "clippy")
                || (s == "fix")
                || (s == "fmt")
                || (s == "test")
                || (s == "t")
                || (s == "bench")
                || (s == "doc")
                || (s == "clean")
                || (s == "dist")
                || (s == "install")
                || (s == "run")
                || (s == "r")
                || (s == "setup")
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
            "test" | "t" => {
                opts.optflag("", "no-fail-fast", "Run all tests regardless of failure");
                opts.optmulti(
                    "",
                    "test-args",
                    "extra arguments to be passed for the test tool being used \
                        (e.g. libtest, compiletest or rustdoc)",
                    "ARGS",
                );
                opts.optmulti(
                    "",
                    "rustc-args",
                    "extra options to pass the compiler when running tests",
                    "ARGS",
                );
                opts.optflag("", "no-doc", "do not run doc tests");
                opts.optflag("", "doc", "only run doc tests");
                opts.optflag("", "bless", "update all stderr/stdout files of failing ui tests");
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
                    "check | build | run",
                );
                opts.optflag(
                    "",
                    "rustfix-coverage",
                    "enable this to generate a Rustfix coverage file, which is saved in \
                        `/<build_base>/rustfix_missing_coverage.txt`",
                );
            }
            "check" | "c" => {
                opts.optflag("", "all-targets", "Check all targets");
            }
            "bench" => {
                opts.optmulti("", "test-args", "extra arguments", "ARGS");
            }
            "clippy" => {
                opts.optflag("", "fix", "automatically apply lint suggestions");
            }
            "doc" => {
                opts.optflag("", "open", "open the docs in a browser");
            }
            "clean" => {
                opts.optflag("", "all", "clean all build artifacts");
            }
            "fmt" => {
                opts.optflag("", "check", "check formatting instead of applying.");
            }
            _ => {}
        };

        // fn usage()
        let usage = |exit_code: i32, opts: &Options, verbose: bool, subcommand_help: &str| -> ! {
            let mut extra_help = String::new();

            // All subcommands except `clean` can have an optional "Available paths" section
            if verbose {
                let config = Config::parse(&["build".to_string()]);
                let build = Build::new(config);

                let maybe_rules_help = Builder::get_help(&build, subcommand.as_str());
                extra_help.push_str(maybe_rules_help.unwrap_or_default().as_str());
            } else if !(subcommand.as_str() == "clean" || subcommand.as_str() == "fmt") {
                extra_help.push_str(
                    format!("Run `./x.py {} -h -v` to see a list of available paths.", subcommand)
                        .as_str(),
                );
            }

            println!("{}", opts.usage(subcommand_help));
            if !extra_help.is_empty() {
                println!("{}", extra_help);
            }
            process::exit(exit_code);
        };

        // Done specifying what options are possible, so do the getopts parsing
        let matches = opts.parse(&args[..]).unwrap_or_else(|e| {
            // Invalid argument/option format
            println!("\n{}\n", e);
            usage(1, &opts, false, &subcommand_help);
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
            "build" | "b" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to compile. For example:

        ./x.py build library/core
        ./x.py build library/core library/proc_macro
        ./x.py build library/std --stage 1

    If no arguments are passed then the complete artifacts for that stage are
    also compiled.

        ./x.py build
        ./x.py build --stage 1

    For a quick build of a usable compiler, you can pass:

        ./x.py build --stage 1 library/test

    This will first build everything once (like `--stage 0` without further
    arguments would), and then use the compiler built in stage 0 to build
    library/test and its dependencies.
    Once this is done, build/$ARCH/stage1 contains a usable compiler.",
                );
            }
            "check" | "c" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to compile. For example:

        ./x.py check library/core
        ./x.py check library/core library/proc_macro

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

        ./x.py clippy library/core
        ./x.py clippy library/core library/proc_macro",
                );
            }
            "fix" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to run `cargo fix` against. For example:

        ./x.py fix library/core
        ./x.py fix library/core library/proc_macro",
                );
            }
            "fmt" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand optionally accepts a `--check` flag which succeeds if formatting is correct and
    fails if it is not. For example:

        ./x.py fmt
        ./x.py fmt --check",
                );
            }
            "test" | "t" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to test directories that
    should be compiled and run. For example:

        ./x.py test src/test/ui
        ./x.py test library/std --test-args hash_map
        ./x.py test library/std --stage 0 --no-doc
        ./x.py test src/test/ui --bless
        ./x.py test src/test/ui --compare-mode nll

    Note that `test src/test/* --stage N` does NOT depend on `build compiler/rustc --stage N`;
    just like `build library/std --stage N` it tests the compiler produced by the previous
    stage.

    Execute tool tests with a tool name argument:

        ./x.py test tidy

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
        ./x.py doc src/doc/book library/std
        ./x.py doc library/std --open

    If no arguments are passed then everything is documented:

        ./x.py doc
        ./x.py doc --stage 1",
                );
            }
            "run" | "r" => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to tools to build and run. For
    example:

        ./x.py run src/tools/expand-yaml-anchors

    At least a tool needs to be called.",
                );
            }
            "setup" => {
                subcommand_help.push_str(&format!(
                    "\n
x.py setup creates a `config.toml` which changes the defaults for x.py itself.

Arguments:
    This subcommand accepts a 'profile' to use for builds. For example:

        ./x.py setup library

    The profile is optional and you will be prompted interactively if it is not given.
    The following profiles are available:

{}",
                    Profile::all_for_help("        ").trim_end()
                ));
            }
            _ => {}
        };
        // Get any optional paths which occur after the subcommand
        let mut paths = matches.free[1..].iter().map(|p| p.into()).collect::<Vec<PathBuf>>();

        let cfg_file = env::var_os("BOOTSTRAP_CONFIG").map(PathBuf::from);
        let verbose = matches.opt_present("verbose");

        // User passed in -h/--help?
        if matches.opt_present("help") {
            usage(0, &opts, verbose, &subcommand_help);
        }

        let cmd = match subcommand.as_str() {
            "build" | "b" => Subcommand::Build { paths },
            "check" | "c" => {
                Subcommand::Check { paths, all_targets: matches.opt_present("all-targets") }
            }
            "clippy" => Subcommand::Clippy { paths, fix: matches.opt_present("fix") },
            "fix" => Subcommand::Fix { paths },
            "test" | "t" => Subcommand::Test {
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
            "bench" => Subcommand::Bench { paths, test_args: matches.opt_strs("test-args") },
            "doc" => Subcommand::Doc { paths, open: matches.opt_present("open") },
            "clean" => {
                if !paths.is_empty() {
                    println!("\nclean does not take a path argument\n");
                    usage(1, &opts, verbose, &subcommand_help);
                }

                Subcommand::Clean { all: matches.opt_present("all") }
            }
            "fmt" => Subcommand::Format { check: matches.opt_present("check") },
            "dist" => Subcommand::Dist { paths },
            "install" => Subcommand::Install { paths },
            "run" | "r" => {
                if paths.is_empty() {
                    println!("\nrun requires at least a path!\n");
                    usage(1, &opts, verbose, &subcommand_help);
                }
                Subcommand::Run { paths }
            }
            "setup" => {
                let profile = if paths.len() > 1 {
                    println!("\nat most one profile can be passed to setup\n");
                    usage(1, &opts, verbose, &subcommand_help)
                } else if let Some(path) = paths.pop() {
                    let profile_string = t!(path.into_os_string().into_string().map_err(
                        |path| format!("{} is not a valid UTF8 string", path.to_string_lossy())
                    ));

                    profile_string.parse().unwrap_or_else(|err| {
                        eprintln!("error: {}", err);
                        eprintln!("help: the available profiles are:");
                        eprint!("{}", Profile::all_for_help("- "));
                        std::process::exit(1);
                    })
                } else {
                    t!(crate::setup::interactive_path())
                };
                Subcommand::Setup { profile }
            }
            _ => {
                usage(1, &opts, verbose, &subcommand_help);
            }
        };

        if let Subcommand::Check { .. } = &cmd {
            if matches.opt_str("stage").is_some() {
                println!("--stage not supported for x.py check, always treated as stage 0");
                process::exit(1);
            }
            if matches.opt_str("keep-stage").is_some()
                || matches.opt_str("keep-stage-std").is_some()
            {
                println!("--keep-stage not supported for x.py check, only one stage available");
                process::exit(1);
            }
        }

        Flags {
            verbose: matches.opt_count("verbose"),
            stage: matches.opt_str("stage").map(|j| j.parse().expect("`stage` should be a number")),
            dry_run: matches.opt_present("dry-run"),
            on_fail: matches.opt_str("on-fail"),
            rustc_error_format: matches.opt_str("error-format"),
            json_output: matches.opt_present("json-output"),
            keep_stage: matches
                .opt_strs("keep-stage")
                .into_iter()
                .map(|j| j.parse().expect("`keep-stage` should be a number"))
                .collect(),
            keep_stage_std: matches
                .opt_strs("keep-stage-std")
                .into_iter()
                .map(|j| j.parse().expect("`keep-stage-std` should be a number"))
                .collect(),
            host: if matches.opt_present("host") {
                Some(
                    split(&matches.opt_strs("host"))
                        .into_iter()
                        .map(|x| TargetSelection::from_user(&x))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            },
            target: if matches.opt_present("target") {
                Some(
                    split(&matches.opt_strs("target"))
                        .into_iter()
                        .map(|x| TargetSelection::from_user(&x))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            },
            config: cfg_file,
            jobs: matches.opt_str("jobs").map(|j| j.parse().expect("`jobs` should be a number")),
            cmd,
            incremental: matches.opt_present("incremental"),
            exclude: split(&matches.opt_strs("exclude"))
                .into_iter()
                .map(|p| p.into())
                .collect::<Vec<_>>(),
            include_default_paths: matches.opt_present("include-default-paths"),
            deny_warnings: parse_deny_warnings(&matches),
            llvm_skip_rebuild: matches.opt_str("llvm-skip-rebuild").map(|s| s.to_lowercase()).map(
                |s| s.parse::<bool>().expect("`llvm-skip-rebuild` should be either true or false"),
            ),
        }
    }
}

impl Subcommand {
    pub fn test_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref test_args, .. } | Subcommand::Bench { ref test_args, .. } => {
                test_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn rustc_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref rustc_args, .. } => {
                rustc_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
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
            Subcommand::Test { ref compare_mode, .. } => compare_mode.as_ref().map(|s| &s[..]),
            _ => None,
        }
    }

    pub fn pass(&self) -> Option<&str> {
        match *self {
            Subcommand::Test { ref pass, .. } => pass.as_ref().map(|s| &s[..]),
            _ => None,
        }
    }

    pub fn open(&self) -> bool {
        match *self {
            Subcommand::Doc { open, .. } => open,
            _ => false,
        }
    }
}

fn split(s: &[String]) -> Vec<String> {
    s.iter().flat_map(|s| s.split(',')).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

fn parse_deny_warnings(matches: &getopts::Matches) -> Option<bool> {
    match matches.opt_str("warnings").as_deref() {
        Some("deny") => Some(true),
        Some("warn") => Some(false),
        Some(value) => {
            eprintln!(r#"invalid value for --warnings: {:?}, expected "warn" or "deny""#, value,);
            process::exit(1);
        }
        None => None,
    }
}
