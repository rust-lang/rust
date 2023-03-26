//! Command-line interface of the rustbuild build system.
//!
//! This module implements the command-line parsing of the build system which
//! has various flags to configure how it's run.

use std::path::PathBuf;

use getopts::Options;

use crate::builder::{Builder, Kind};
use crate::config::{Config, TargetSelection};
use crate::setup::Profile;
use crate::util::t;
use crate::{Build, DocTests};

#[derive(Copy, Clone)]
pub enum Color {
    Always,
    Never,
    Auto,
}

impl Default for Color {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for Color {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            "auto" => Ok(Self::Auto),
            _ => Err(()),
        }
    }
}

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
    pub build_dir: Option<PathBuf>,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
    pub exclude: Vec<PathBuf>,
    pub include_default_paths: bool,
    pub rustc_error_format: Option<String>,
    pub json_output: bool,
    pub dry_run: bool,
    pub color: Color,

    // This overrides the deny-warnings configuration option,
    // which passes -Dwarnings to the compiler invocations.
    //
    // true => deny, false => warn
    pub deny_warnings: Option<bool>,

    pub rust_profile_use: Option<String>,
    pub rust_profile_generate: Option<String>,

    pub llvm_profile_use: Option<String>,
    // LLVM doesn't support a custom location for generating profile
    // information.
    //
    // llvm_out/build/profiles/ is the location this writes to.
    pub llvm_profile_generate: bool,
    pub llvm_bolt_profile_generate: bool,
    pub llvm_bolt_profile_use: Option<String>,

    /// Arguments appearing after `--` to be forwarded to tools,
    /// e.g. `--fix-broken` or test arguments.
    pub free_args: Option<Vec<String>>,
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub enum Subcommand {
    Build {
        paths: Vec<PathBuf>,
    },
    Check {
        paths: Vec<PathBuf>,
    },
    Clippy {
        fix: bool,
        paths: Vec<PathBuf>,
        clippy_lint_allow: Vec<String>,
        clippy_lint_deny: Vec<String>,
        clippy_lint_warn: Vec<String>,
        clippy_lint_forbid: Vec<String>,
    },
    Fix {
        paths: Vec<PathBuf>,
    },
    Format {
        paths: Vec<PathBuf>,
        check: bool,
    },
    Doc {
        paths: Vec<PathBuf>,
        open: bool,
        json: bool,
    },
    Test {
        paths: Vec<PathBuf>,
        /// Whether to automatically update stderr/stdout files
        bless: bool,
        force_rerun: bool,
        compare_mode: Option<String>,
        pass: Option<String>,
        run: Option<String>,
        test_args: Vec<String>,
        rustc_args: Vec<String>,
        fail_fast: bool,
        doc_tests: DocTests,
        rustfix_coverage: bool,
        only_modified: bool,
    },
    Bench {
        paths: Vec<PathBuf>,
        test_args: Vec<String>,
    },
    Clean {
        paths: Vec<PathBuf>,
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
        args: Vec<String>,
    },
    Setup {
        profile: Option<PathBuf>,
    },
}

impl Default for Subcommand {
    fn default() -> Subcommand {
        Subcommand::Build { paths: vec![PathBuf::from("nowhere")] }
    }
}

impl Flags {
    pub fn parse(args: &[String]) -> Flags {
        let (args, free_args) = if let Some(pos) = args.iter().position(|s| s == "--") {
            let (args, free) = args.split_at(pos);
            (args, Some(free[1..].to_vec()))
        } else {
            (args, None)
        };
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
    doc, d      Build documentation
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
        opts.optopt(
            "",
            "build-dir",
            "Build directory, overrides `build.build-dir` in `config.toml`",
            "DIR",
        );
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
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
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
        opts.optopt("", "color", "whether to use color in cargo and rustc output", "STYLE");
        opts.optopt(
            "",
            "rust-profile-generate",
            "generate PGO profile with rustc build",
            "PROFILE",
        );
        opts.optopt("", "rust-profile-use", "use PGO profile for rustc build", "PROFILE");
        opts.optflag("", "llvm-profile-generate", "generate PGO profile with llvm built for rustc");
        opts.optopt("", "llvm-profile-use", "use PGO profile for llvm build", "PROFILE");
        opts.optmulti("A", "", "allow certain clippy lints", "OPT");
        opts.optmulti("D", "", "deny certain clippy lints", "OPT");
        opts.optmulti("W", "", "warn about certain clippy lints", "OPT");
        opts.optmulti("F", "", "forbid certain clippy lints", "OPT");
        opts.optflag("", "llvm-bolt-profile-generate", "generate BOLT profile for LLVM build");
        opts.optopt("", "llvm-bolt-profile-use", "use BOLT profile for LLVM build", "PROFILE");

        // We can't use getopt to parse the options until we have completed specifying which
        // options are valid, but under the current implementation, some options are conditional on
        // the subcommand. Therefore we must manually identify the subcommand first, so that we can
        // complete the definition of the options.  Then we can use the getopt::Matches object from
        // there on out.
        let subcommand = match args.iter().find_map(|s| Kind::parse(&s)) {
            Some(s) => s,
            None => {
                // No or an invalid subcommand -- show the general usage and subcommand help
                // An exit code will be 0 when no subcommand is given, and 1 in case of an invalid
                // subcommand.
                println!("{}\n", subcommand_help);
                let exit_code = if args.is_empty() { 0 } else { 1 };
                crate::detail_exit(exit_code);
            }
        };

        // Some subcommands get extra options
        match subcommand {
            Kind::Test => {
                opts.optflag("", "no-fail-fast", "Run all tests regardless of failure");
                opts.optmulti("", "skip", "skips tests matching SUBSTRING, if supported by test tool. May be passed multiple times", "SUBSTRING");
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
                opts.optflag("", "force-rerun", "rerun tests even if the inputs are unchanged");
                opts.optflag("", "only-modified", "only run tests that result has been changed");
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
                opts.optopt("", "run", "whether to execute run-* tests", "auto | always | never");
                opts.optflag(
                    "",
                    "rustfix-coverage",
                    "enable this to generate a Rustfix coverage file, which is saved in \
                        `/<build_base>/rustfix_missing_coverage.txt`",
                );
            }
            Kind::Check => {
                opts.optflag("", "all-targets", "Check all targets");
            }
            Kind::Bench => {
                opts.optmulti("", "test-args", "extra arguments", "ARGS");
            }
            Kind::Clippy => {
                opts.optflag("", "fix", "automatically apply lint suggestions");
            }
            Kind::Doc => {
                opts.optflag("", "open", "open the docs in a browser");
                opts.optflag(
                    "",
                    "json",
                    "render the documentation in JSON format in addition to the usual HTML format",
                );
            }
            Kind::Clean => {
                opts.optflag("", "all", "clean all build artifacts");
            }
            Kind::Format => {
                opts.optflag("", "check", "check formatting instead of applying.");
            }
            Kind::Run => {
                opts.optmulti("", "args", "arguments for the tool", "ARGS");
            }
            _ => {}
        };

        // fn usage()
        let usage = |exit_code: i32, opts: &Options, verbose: bool, subcommand_help: &str| -> ! {
            println!("{}", opts.usage(subcommand_help));
            if verbose {
                // We have an unfortunate situation here: some Steps use `builder.in_tree_crates` to determine their paths.
                // To determine those crates, we need to run `cargo metadata`, which means we need all submodules to be checked out.
                // That takes a while to run, so only do it when paths were explicitly requested, not on all CLI errors.
                // `Build::new` won't load submodules for the `setup` command.
                let cmd = if verbose {
                    println!("note: updating submodules before printing available paths");
                    "build"
                } else {
                    "setup"
                };
                let config = Config::parse(&[cmd.to_string()]);
                let build = Build::new(config);
                let paths = Builder::get_help(&build, subcommand);

                if let Some(s) = paths {
                    println!("{}", s);
                } else {
                    panic!("No paths available for subcommand `{}`", subcommand.as_str());
                }
            } else {
                println!(
                    "Run `./x.py {} -h -v` to see a list of available paths.",
                    subcommand.as_str()
                );
            }
            crate::detail_exit(exit_code);
        };

        // Done specifying what options are possible, so do the getopts parsing
        let matches = opts.parse(args).unwrap_or_else(|e| {
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
        match matches.free.get(0).and_then(|s| Kind::parse(&s)) {
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
            eprintln!("{}\n", subcommand_help);
            eprintln!(
                "Sorry, I couldn't figure out which subcommand you were trying to specify.\n\
                 You may need to move some options to after the subcommand.\n"
            );
            crate::detail_exit(1);
        }
        // Extra help text for some commands
        match subcommand {
            Kind::Build => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to compile. For example, for a quick build of a usable
    compiler:

        ./x.py build --stage 1 library/std

    This will build a compiler and standard library from the local source code.
    Once this is done, build/$ARCH/stage1 contains a usable compiler.

    If no arguments are passed then the default artifacts for that stage are
    compiled. For example:

        ./x.py build --stage 0
        ./x.py build ",
                );
            }
            Kind::Check => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to compile. For example:

        ./x.py check library/std

    If no arguments are passed then many artifacts are checked.",
                );
            }
            Kind::Clippy => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to run clippy against. For example:

        ./x.py clippy library/core
        ./x.py clippy library/core library/proc_macro",
                );
            }
            Kind::Fix => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories to the crates
    and/or artifacts to run `cargo fix` against. For example:

        ./x.py fix library/core
        ./x.py fix library/core library/proc_macro",
                );
            }
            Kind::Format => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand optionally accepts a `--check` flag which succeeds if formatting is correct and
    fails if it is not. For example:

        ./x.py fmt
        ./x.py fmt --check",
                );
            }
            Kind::Test => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to test directories that
    should be compiled and run. For example:

        ./x.py test tests/ui
        ./x.py test library/std --test-args hash_map
        ./x.py test library/std --stage 0 --no-doc
        ./x.py test tests/ui --bless
        ./x.py test tests/ui --compare-mode chalk

    Note that `test tests/* --stage N` does NOT depend on `build compiler/rustc --stage N`;
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
            Kind::Doc => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to directories of documentation
    to build. For example:

        ./x.py doc src/doc/book
        ./x.py doc src/doc/nomicon
        ./x.py doc src/doc/book library/std
        ./x.py doc library/std --json
        ./x.py doc library/std --open

    If no arguments are passed then everything is documented:

        ./x.py doc
        ./x.py doc --stage 1",
                );
            }
            Kind::Run => {
                subcommand_help.push_str(
                    "\n
Arguments:
    This subcommand accepts a number of paths to tools to build and run. For
    example:

        ./x.py run src/tools/expand-yaml-anchors

    At least a tool needs to be called.",
                );
            }
            Kind::Setup => {
                subcommand_help.push_str(&format!(
                    "\n
x.py setup creates a `config.toml` which changes the defaults for x.py itself,
as well as setting up a git pre-push hook, VS code config and toolchain link.

Arguments:
    This subcommand accepts a 'profile' to use for builds. For example:

        ./x.py setup library

    The profile is optional and you will be prompted interactively if it is not given.
    The following profiles are available:

{}

    To only set up the git hook, VS code or toolchain link, you may use
        ./x.py setup hook
        ./x.py setup vscode
        ./x.py setup link
",
                    Profile::all_for_help("        ").trim_end()
                ));
            }
            Kind::Bench | Kind::Clean | Kind::Dist | Kind::Install => {}
        };
        // Get any optional paths which occur after the subcommand
        let mut paths = matches.free[1..].iter().map(|p| p.into()).collect::<Vec<PathBuf>>();

        let verbose = matches.opt_present("verbose");

        // User passed in -h/--help?
        if matches.opt_present("help") {
            usage(0, &opts, verbose, &subcommand_help);
        }

        let cmd = match subcommand {
            Kind::Build => Subcommand::Build { paths },
            Kind::Check => {
                if matches.opt_present("all-targets") {
                    println!(
                        "Warning: --all-targets is now on by default and does not need to be passed explicitly."
                    );
                }
                Subcommand::Check { paths }
            }
            Kind::Clippy => Subcommand::Clippy {
                paths,
                fix: matches.opt_present("fix"),
                clippy_lint_allow: matches.opt_strs("A"),
                clippy_lint_warn: matches.opt_strs("W"),
                clippy_lint_deny: matches.opt_strs("D"),
                clippy_lint_forbid: matches.opt_strs("F"),
            },
            Kind::Fix => Subcommand::Fix { paths },
            Kind::Test => Subcommand::Test {
                paths,
                bless: matches.opt_present("bless"),
                force_rerun: matches.opt_present("force-rerun"),
                compare_mode: matches.opt_str("compare-mode"),
                pass: matches.opt_str("pass"),
                run: matches.opt_str("run"),
                test_args: matches.opt_strs("test-args"),
                rustc_args: matches.opt_strs("rustc-args"),
                fail_fast: !matches.opt_present("no-fail-fast"),
                rustfix_coverage: matches.opt_present("rustfix-coverage"),
                only_modified: matches.opt_present("only-modified"),
                doc_tests: if matches.opt_present("doc") {
                    DocTests::Only
                } else if matches.opt_present("no-doc") {
                    DocTests::No
                } else {
                    DocTests::Yes
                },
            },
            Kind::Bench => Subcommand::Bench { paths, test_args: matches.opt_strs("test-args") },
            Kind::Doc => Subcommand::Doc {
                paths,
                open: matches.opt_present("open"),
                json: matches.opt_present("json"),
            },
            Kind::Clean => Subcommand::Clean { all: matches.opt_present("all"), paths },
            Kind::Format => Subcommand::Format { check: matches.opt_present("check"), paths },
            Kind::Dist => Subcommand::Dist { paths },
            Kind::Install => Subcommand::Install { paths },
            Kind::Run => {
                if paths.is_empty() {
                    println!("\nrun requires at least a path!\n");
                    usage(1, &opts, verbose, &subcommand_help);
                }
                Subcommand::Run { paths, args: matches.opt_strs("args") }
            }
            Kind::Setup => {
                let profile = if paths.len() > 1 {
                    eprintln!("\nerror: At most one option can be passed to setup\n");
                    usage(1, &opts, verbose, &subcommand_help)
                } else if let Some(path) = paths.pop() {
                    let profile_string = t!(path.into_os_string().into_string().map_err(
                        |path| format!("{} is not a valid UTF8 string", path.to_string_lossy())
                    ));

                    let profile = profile_string.parse().unwrap_or_else(|err| {
                        eprintln!("error: {}", err);
                        eprintln!("help: the available profiles are:");
                        eprint!("{}", Profile::all_for_help("- "));
                        crate::detail_exit(1);
                    });
                    Some(profile)
                } else {
                    None
                };
                Subcommand::Setup { profile }
            }
        };

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
            config: matches.opt_str("config").map(PathBuf::from),
            build_dir: matches.opt_str("build-dir").map(PathBuf::from),
            jobs: matches.opt_str("jobs").map(|j| j.parse().expect("`jobs` should be a number")),
            cmd,
            incremental: matches.opt_present("incremental"),
            exclude: split(&matches.opt_strs("exclude"))
                .into_iter()
                .map(|p| p.into())
                .collect::<Vec<_>>(),
            include_default_paths: matches.opt_present("include-default-paths"),
            deny_warnings: parse_deny_warnings(&matches),
            color: matches
                .opt_get_default("color", Color::Auto)
                .expect("`color` should be `always`, `never`, or `auto`"),
            rust_profile_use: matches.opt_str("rust-profile-use"),
            rust_profile_generate: matches.opt_str("rust-profile-generate"),
            llvm_profile_use: matches.opt_str("llvm-profile-use"),
            llvm_profile_generate: matches.opt_present("llvm-profile-generate"),
            llvm_bolt_profile_generate: matches.opt_present("llvm-bolt-profile-generate"),
            llvm_bolt_profile_use: matches.opt_str("llvm-bolt-profile-use"),
            free_args,
        }
    }
}

impl Subcommand {
    pub fn kind(&self) -> Kind {
        match self {
            Subcommand::Bench { .. } => Kind::Bench,
            Subcommand::Build { .. } => Kind::Build,
            Subcommand::Check { .. } => Kind::Check,
            Subcommand::Clippy { .. } => Kind::Clippy,
            Subcommand::Doc { .. } => Kind::Doc,
            Subcommand::Fix { .. } => Kind::Fix,
            Subcommand::Format { .. } => Kind::Format,
            Subcommand::Test { .. } => Kind::Test,
            Subcommand::Clean { .. } => Kind::Clean,
            Subcommand::Dist { .. } => Kind::Dist,
            Subcommand::Install { .. } => Kind::Install,
            Subcommand::Run { .. } => Kind::Run,
            Subcommand::Setup { .. } => Kind::Setup,
        }
    }

    pub fn test_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref test_args, .. } | Subcommand::Bench { ref test_args, .. } => {
                test_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
        }
    }

    pub fn rustc_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref rustc_args, .. } => {
                rustc_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
        }
    }

    pub fn args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Run { ref args, .. } => {
                args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
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

    pub fn only_modified(&self) -> bool {
        match *self {
            Subcommand::Test { only_modified, .. } => only_modified,
            _ => false,
        }
    }

    pub fn force_rerun(&self) -> bool {
        match *self {
            Subcommand::Test { force_rerun, .. } => force_rerun,
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

    pub fn run(&self) -> Option<&str> {
        match *self {
            Subcommand::Test { ref run, .. } => run.as_ref().map(|s| &s[..]),
            _ => None,
        }
    }

    pub fn open(&self) -> bool {
        match *self {
            Subcommand::Doc { open, .. } => open,
            _ => false,
        }
    }

    pub fn json(&self) -> bool {
        match *self {
            Subcommand::Doc { json, .. } => json,
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
            crate::detail_exit(1);
        }
        None => None,
    }
}
