//! Command-line interface of the bootstrap build system.
//!
//! This module implements the command-line parsing of the build system which
//! has various flags to configure how it's run.

use std::path::{Path, PathBuf};

use clap::{CommandFactory, Parser, ValueEnum};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::core::build_steps::perf::PerfArgs;
use crate::core::build_steps::setup::Profile;
use crate::core::builder::{Builder, Kind};
use crate::core::config::Config;
use crate::core::config::target_selection::{TargetSelectionList, target_selection_list};
use crate::{Build, DocTests};

#[derive(Copy, Clone, Default, Debug, ValueEnum)]
pub enum Color {
    Always,
    Never,
    #[default]
    Auto,
}

/// Whether to deny warnings, emit them as warnings, or use the default behavior
#[derive(Copy, Clone, Default, Debug, ValueEnum)]
pub enum Warnings {
    Deny,
    Warn,
    #[default]
    Default,
}

/// Deserialized version of all flags for this compile.
#[derive(Debug, Parser)]
#[command(
    override_usage = "x.py <subcommand> [options] [<paths>...]",
    disable_help_subcommand(true),
    about = "",
    next_line_help(false)
)]
pub struct Flags {
    #[command(subcommand)]
    pub cmd: Subcommand,

    #[arg(global = true, short, long, action = clap::ArgAction::Count)]
    /// use verbose output (-vv for very verbose)
    pub verbose: u8, // each extra -v after the first is passed to Cargo
    #[arg(global = true, short, long)]
    /// use incremental compilation
    pub incremental: bool,
    #[arg(global = true, long, value_hint = clap::ValueHint::FilePath, value_name = "FILE")]
    /// TOML configuration file for build
    pub config: Option<PathBuf>,
    #[arg(global = true, long, value_hint = clap::ValueHint::DirPath, value_name = "DIR")]
    /// Build directory, overrides `build.build-dir` in `bootstrap.toml`
    pub build_dir: Option<PathBuf>,

    #[arg(global = true, long, value_hint = clap::ValueHint::Other, value_name = "BUILD")]
    /// host target of the stage0 compiler
    pub build: Option<String>,

    #[arg(global = true, long, value_hint = clap::ValueHint::Other, value_name = "HOST", value_parser = target_selection_list)]
    /// host targets to build
    pub host: Option<TargetSelectionList>,

    #[arg(global = true, long, value_hint = clap::ValueHint::Other, value_name = "TARGET", value_parser = target_selection_list)]
    /// target targets to build
    pub target: Option<TargetSelectionList>,

    #[arg(global = true, long, value_name = "PATH")]
    /// build paths to exclude
    pub exclude: Vec<PathBuf>, // keeping for client backward compatibility
    #[arg(global = true, long, value_name = "PATH")]
    /// build paths to skip
    pub skip: Vec<PathBuf>,
    #[arg(global = true, long)]
    /// include default paths in addition to the provided ones
    pub include_default_paths: bool,

    /// rustc error format
    #[arg(global = true, value_hint = clap::ValueHint::Other, long)]
    pub rustc_error_format: Option<String>,

    #[arg(global = true, long, value_hint = clap::ValueHint::CommandString, value_name = "CMD")]
    /// command to run on failure
    pub on_fail: Option<String>,
    #[arg(global = true, long)]
    /// dry run; don't build anything
    pub dry_run: bool,
    /// Indicates whether to dump the work done from bootstrap shims
    #[arg(global = true, long)]
    pub dump_bootstrap_shims: bool,
    #[arg(global = true, value_hint = clap::ValueHint::Other, long, value_name = "N")]
    /// stage to build (indicates compiler to use/test, e.g., stage 0 uses the
    /// bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)
    pub stage: Option<u32>,

    #[arg(global = true, value_hint = clap::ValueHint::Other, long, value_name = "N")]
    /// stage(s) to keep without recompiling
    /// (pass multiple times to keep e.g., both stages 0 and 1)
    pub keep_stage: Vec<u32>,
    #[arg(global = true, value_hint = clap::ValueHint::Other, long, value_name = "N")]
    /// stage(s) of the standard library to keep without recompiling
    /// (pass multiple times to keep e.g., both stages 0 and 1)
    pub keep_stage_std: Vec<u32>,
    #[arg(global = true, long, value_hint = clap::ValueHint::DirPath, value_name = "DIR")]
    /// path to the root of the rust checkout
    pub src: Option<PathBuf>,

    #[arg(
        global = true,
        short,
        long,
        value_hint = clap::ValueHint::Other,
        value_name = "JOBS"
    )]
    /// number of jobs to run in parallel
    pub jobs: Option<u32>,
    // This overrides the deny-warnings configuration option,
    // which passes -Dwarnings to the compiler invocations.
    #[arg(global = true, long)]
    #[arg(value_enum, default_value_t=Warnings::Default, value_name = "deny|warn")]
    /// if value is deny, will deny warnings
    /// if value is warn, will emit warnings
    /// otherwise, use the default configured behaviour
    pub warnings: Warnings,

    #[arg(global = true, long)]
    /// use message-format=json
    pub json_output: bool,

    #[arg(global = true, long, value_name = "STYLE")]
    #[arg(value_enum, default_value_t = Color::Auto)]
    /// whether to use color in cargo and rustc output
    pub color: Color,

    #[arg(global = true, long)]
    /// Bootstrap uses this value to decide whether it should bypass locking the build process.
    /// This is rarely needed (e.g., compiling the std library for different targets in parallel).
    ///
    /// Unless you know exactly what you are doing, you probably don't need this.
    pub bypass_bootstrap_lock: bool,

    /// generate PGO profile with rustc build
    #[arg(global = true, value_hint = clap::ValueHint::FilePath, long, value_name = "PROFILE")]
    pub rust_profile_generate: Option<String>,
    /// use PGO profile for rustc build
    #[arg(global = true, value_hint = clap::ValueHint::FilePath, long, value_name = "PROFILE")]
    pub rust_profile_use: Option<String>,
    /// use PGO profile for LLVM build
    #[arg(global = true, value_hint = clap::ValueHint::FilePath, long, value_name = "PROFILE")]
    pub llvm_profile_use: Option<String>,
    // LLVM doesn't support a custom location for generating profile
    // information.
    //
    // llvm_out/build/profiles/ is the location this writes to.
    /// generate PGO profile with llvm built for rustc
    #[arg(global = true, long)]
    pub llvm_profile_generate: bool,
    /// Enable BOLT link flags
    #[arg(global = true, long)]
    pub enable_bolt_settings: bool,
    /// Skip stage0 compiler validation
    #[arg(global = true, long)]
    pub skip_stage0_validation: bool,
    /// Additional reproducible artifacts that should be added to the reproducible artifacts archive.
    #[arg(global = true, long)]
    pub reproducible_artifact: Vec<String>,
    #[arg(global = true)]
    /// paths for the subcommand
    pub paths: Vec<PathBuf>,
    /// override options in bootstrap.toml
    #[arg(global = true, value_hint = clap::ValueHint::Other, long, value_name = "section.option=value")]
    pub set: Vec<String>,
    /// arguments passed to subcommands
    #[arg(global = true, last(true), value_name = "ARGS")]
    pub free_args: Vec<String>,
    /// Make bootstrap to behave as it's running on the CI environment or not.
    #[arg(global = true, long, value_name = "bool")]
    pub ci: Option<bool>,
    /// Skip checking the standard library if `rust.download-rustc` isn't available.
    /// This is mostly for RA as building the stage1 compiler to check the library tree
    /// on each code change might be too much for some computers.
    #[arg(global = true, long)]
    pub skip_std_check_if_no_download_rustc: bool,
}

impl Flags {
    /// Check if `<cmd> -h -v` was passed.
    /// If yes, print the available paths and return `true`.
    pub fn try_parse_verbose_help(args: &[String]) -> bool {
        // We need to check for `<cmd> -h -v`, in which case we list the paths
        #[derive(Parser)]
        #[command(disable_help_flag(true))]
        struct HelpVerboseOnly {
            #[arg(short, long)]
            help: bool,
            #[arg(global = true, short, long, action = clap::ArgAction::Count)]
            pub verbose: u8,
            #[arg(value_enum)]
            cmd: Kind,
        }
        if let Ok(HelpVerboseOnly { help: true, verbose: 1.., cmd: subcommand }) =
            HelpVerboseOnly::try_parse_from(normalize_args(args))
        {
            println!("NOTE: updating submodules before printing available paths");
            let flags = Self::parse(&[String::from("build")]);
            let config = Config::parse(flags);
            let build = Build::new(config);
            let paths = Builder::get_help(&build, subcommand);
            if let Some(s) = paths {
                println!("{s}");
            } else {
                panic!("No paths available for subcommand `{}`", subcommand.as_str());
            }
            true
        } else {
            false
        }
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(level = "trace", name = "Flags::parse", skip_all, fields(args = ?args))
    )]
    pub fn parse(args: &[String]) -> Self {
        Flags::parse_from(normalize_args(args))
    }
}

fn normalize_args(args: &[String]) -> Vec<String> {
    let first = String::from("x.py");
    let it = std::iter::once(first).chain(args.iter().cloned());
    it.collect()
}

#[derive(Debug, Clone, Default, clap::Subcommand)]
pub enum Subcommand {
    #[command(aliases = ["b"], long_about = "\n
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
            ./x.py build ")]
    /// Compile either the compiler or libraries
    #[default]
    Build,
    #[command(aliases = ["c"], long_about = "\n
    Arguments:
        This subcommand accepts a number of paths to directories to the crates
        and/or artifacts to compile. For example:
            ./x.py check library/std
        If no arguments are passed then many artifacts are checked.")]
    /// Compile either the compiler or libraries, using cargo check
    Check {
        #[arg(long)]
        /// Check all targets
        all_targets: bool,
    },
    /// Run Clippy (uses rustup/cargo-installed clippy binary)
    #[command(long_about = "\n
    Arguments:
        This subcommand accepts a number of paths to directories to the crates
        and/or artifacts to run clippy against. For example:
            ./x.py clippy library/core
            ./x.py clippy library/core library/proc_macro")]
    Clippy {
        #[arg(long)]
        fix: bool,
        #[arg(long, requires = "fix")]
        allow_dirty: bool,
        #[arg(long, requires = "fix")]
        allow_staged: bool,
        /// clippy lints to allow
        #[arg(global = true, short = 'A', action = clap::ArgAction::Append, value_name = "LINT")]
        allow: Vec<String>,
        /// clippy lints to deny
        #[arg(global = true, short = 'D', action = clap::ArgAction::Append, value_name = "LINT")]
        deny: Vec<String>,
        /// clippy lints to warn on
        #[arg(global = true, short = 'W', action = clap::ArgAction::Append, value_name = "LINT")]
        warn: Vec<String>,
        /// clippy lints to forbid
        #[arg(global = true, short = 'F', action = clap::ArgAction::Append, value_name = "LINT")]
        forbid: Vec<String>,
    },
    /// Run cargo fix
    #[command(long_about = "\n
    Arguments:
        This subcommand accepts a number of paths to directories to the crates
        and/or artifacts to run `cargo fix` against. For example:
            ./x.py fix library/core
            ./x.py fix library/core library/proc_macro")]
    Fix,
    #[command(
        name = "fmt",
        long_about = "\n
    Arguments:
        This subcommand optionally accepts a `--check` flag which succeeds if
        formatting is correct and fails if it is not. For example:
            ./x.py fmt
            ./x.py fmt --check"
    )]
    /// Run rustfmt
    Format {
        /// check formatting instead of applying
        #[arg(long)]
        check: bool,

        /// apply to all appropriate files, not just those that have been modified
        #[arg(long)]
        all: bool,
    },
    #[command(aliases = ["d"], long_about = "\n
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
            ./x.py doc --stage 1")]
    /// Build documentation
    Doc {
        #[arg(long)]
        /// open the docs in a browser
        open: bool,
        #[arg(long)]
        /// render the documentation in JSON format in addition to the usual HTML format
        json: bool,
    },
    #[command(aliases = ["t"], long_about = "\n
    Arguments:
        This subcommand accepts a number of paths to test directories that
        should be compiled and run. For example:
            ./x.py test tests/ui
            ./x.py test library/std --test-args hash_map
            ./x.py test library/std --stage 0 --no-doc
            ./x.py test tests/ui --bless
            ./x.py test tests/ui --compare-mode next-solver
        Note that `test tests/* --stage N` does NOT depend on `build compiler/rustc --stage N`;
        just like `build library/std --stage N` it tests the compiler produced by the previous
        stage.
        Execute tool tests with a tool name argument:
            ./x.py test tidy
        If no arguments are passed then the complete artifacts for that stage are
        compiled and tested.
            ./x.py test
            ./x.py test --stage 1")]
    /// Build and run some test suites
    Test {
        #[arg(long)]
        /// run all tests regardless of failure
        no_fail_fast: bool,
        #[arg(long, value_name = "ARGS", allow_hyphen_values(true))]
        /// extra arguments to be passed for the test tool being used
        /// (e.g. libtest, compiletest or rustdoc)
        test_args: Vec<String>,
        /// extra options to pass the compiler when running compiletest tests
        #[arg(long, value_name = "ARGS", allow_hyphen_values(true))]
        compiletest_rustc_args: Vec<String>,
        #[arg(long)]
        /// do not run doc tests
        no_doc: bool,
        #[arg(long)]
        /// only run doc tests
        doc: bool,
        #[arg(long)]
        /// whether to automatically update stderr/stdout files
        bless: bool,
        #[arg(long)]
        /// comma-separated list of other files types to check (accepts py, py:lint,
        /// py:fmt, shell)
        extra_checks: Option<String>,
        #[arg(long)]
        /// rerun tests even if the inputs are unchanged
        force_rerun: bool,
        #[arg(long)]
        /// only run tests that result has been changed
        only_modified: bool,
        #[arg(long, value_name = "COMPARE MODE")]
        /// mode describing what file the actual ui output will be compared to
        compare_mode: Option<String>,
        #[arg(long, value_name = "check | build | run")]
        /// force {check,build,run}-pass tests to this mode.
        pass: Option<String>,
        #[arg(long, value_name = "auto | always | never")]
        /// whether to execute run-* tests
        run: Option<String>,
        #[arg(long)]
        /// enable this to generate a Rustfix coverage file, which is saved in
        /// `/<build_base>/rustfix_missing_coverage.txt`
        rustfix_coverage: bool,
        #[arg(long)]
        /// don't capture stdout/stderr of tests
        no_capture: bool,
    },
    /// Build and run some test suites *in Miri*
    Miri {
        #[arg(long)]
        /// run all tests regardless of failure
        no_fail_fast: bool,
        #[arg(long, value_name = "ARGS", allow_hyphen_values(true))]
        /// extra arguments to be passed for the test tool being used
        /// (e.g. libtest, compiletest or rustdoc)
        test_args: Vec<String>,
        #[arg(long)]
        /// do not run doc tests
        no_doc: bool,
        #[arg(long)]
        /// only run doc tests
        doc: bool,
    },
    /// Build and run some benchmarks
    Bench {
        #[arg(long, allow_hyphen_values(true))]
        test_args: Vec<String>,
    },
    /// Clean out build directories
    Clean {
        #[arg(long)]
        /// Clean the entire build directory (not used by default)
        all: bool,
        #[arg(long, value_name = "N")]
        /// Clean a specific stage without touching other artifacts. By default, every stage is cleaned if this option is not used.
        stage: Option<u32>,
    },
    /// Build distribution artifacts
    Dist,
    /// Install distribution artifacts
    Install,
    #[command(aliases = ["r"], long_about = "\n
    Arguments:
        This subcommand accepts a number of paths to tools to build and run. For
        example:
            ./x.py run src/tools/bump-stage0
        At least a tool needs to be called.")]
    /// Run tools contained in this repository
    Run {
        /// arguments for the tool
        #[arg(long, allow_hyphen_values(true))]
        args: Vec<String>,
    },
    /// Set up the environment for development
    #[command(long_about = format!(
        "\n
x.py setup creates a `bootstrap.toml` which changes the defaults for x.py itself,
as well as setting up a git pre-push hook, VS Code config and toolchain link.
Arguments:
    This subcommand accepts a 'profile' to use for builds. For example:
        ./x.py setup library
    The profile is optional and you will be prompted interactively if it is not given.
    The following profiles are available:
{}
    To only set up the git hook, editor config or toolchain link, you may use
        ./x.py setup hook
        ./x.py setup editor
        ./x.py setup link", Profile::all_for_help("        ").trim_end()))]
    Setup {
        /// Either the profile for `bootstrap.toml` or another setup action.
        /// May be omitted to set up interactively
        #[arg(value_name = "<PROFILE>|hook|editor|link")]
        profile: Option<PathBuf>,
    },
    /// Suggest a subset of tests to run, based on modified files
    #[command(long_about = "\n")]
    Suggest {
        /// run suggested tests
        #[arg(long)]
        run: bool,
    },
    /// Vendor dependencies
    Vendor {
        /// Additional `Cargo.toml` to sync and vendor
        #[arg(long)]
        sync: Vec<PathBuf>,
        /// Always include version in subdir name
        #[arg(long)]
        versioned_dirs: bool,
    },
    /// Perform profiling and benchmarking of the compiler using `rustc-perf`.
    Perf(PerfArgs),
}

impl Subcommand {
    pub fn kind(&self) -> Kind {
        match self {
            Subcommand::Bench { .. } => Kind::Bench,
            Subcommand::Build => Kind::Build,
            Subcommand::Check { .. } => Kind::Check,
            Subcommand::Clippy { .. } => Kind::Clippy,
            Subcommand::Doc { .. } => Kind::Doc,
            Subcommand::Fix => Kind::Fix,
            Subcommand::Format { .. } => Kind::Format,
            Subcommand::Test { .. } => Kind::Test,
            Subcommand::Miri { .. } => Kind::Miri,
            Subcommand::Clean { .. } => Kind::Clean,
            Subcommand::Dist => Kind::Dist,
            Subcommand::Install => Kind::Install,
            Subcommand::Run { .. } => Kind::Run,
            Subcommand::Setup { .. } => Kind::Setup,
            Subcommand::Suggest { .. } => Kind::Suggest,
            Subcommand::Vendor { .. } => Kind::Vendor,
            Subcommand::Perf { .. } => Kind::Perf,
        }
    }

    pub fn compiletest_rustc_args(&self) -> Vec<&str> {
        match *self {
            Subcommand::Test { ref compiletest_rustc_args, .. } => {
                compiletest_rustc_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
        }
    }

    pub fn fail_fast(&self) -> bool {
        match *self {
            Subcommand::Test { no_fail_fast, .. } | Subcommand::Miri { no_fail_fast, .. } => {
                !no_fail_fast
            }
            _ => false,
        }
    }

    pub fn doc_tests(&self) -> DocTests {
        match *self {
            Subcommand::Test { doc, no_doc, .. } | Subcommand::Miri { no_doc, doc, .. } => {
                if doc {
                    DocTests::Only
                } else if no_doc {
                    DocTests::No
                } else {
                    DocTests::Yes
                }
            }
            _ => DocTests::Yes,
        }
    }

    pub fn bless(&self) -> bool {
        match *self {
            Subcommand::Test { bless, .. } => bless,
            _ => false,
        }
    }

    pub fn extra_checks(&self) -> Option<&str> {
        match *self {
            Subcommand::Test { ref extra_checks, .. } => extra_checks.as_ref().map(String::as_str),
            _ => None,
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

    pub fn no_capture(&self) -> bool {
        match *self {
            Subcommand::Test { no_capture, .. } => no_capture,
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

    pub fn vendor_versioned_dirs(&self) -> bool {
        match *self {
            Subcommand::Vendor { versioned_dirs, .. } => versioned_dirs,
            _ => false,
        }
    }

    pub fn vendor_sync_args(&self) -> Vec<PathBuf> {
        match self {
            Subcommand::Vendor { sync, .. } => sync.clone(),
            _ => vec![],
        }
    }
}

/// Returns the shell completion for a given shell, if the result differs from the current
/// content of `path`. If `path` does not exist, always returns `Some`.
pub fn get_completion<G: clap_complete::Generator>(shell: G, path: &Path) -> Option<String> {
    let mut cmd = Flags::command();
    let current = if !path.exists() {
        String::new()
    } else {
        std::fs::read_to_string(path).unwrap_or_else(|_| {
            eprintln!("couldn't read {}", path.display());
            crate::exit!(1)
        })
    };
    let mut buf = Vec::new();
    let (bin_name, _) = path
        .file_name()
        .expect("path should be a regular file")
        .to_str()
        .expect("file name should be UTF-8")
        .rsplit_once('.')
        .expect("file name should have an extension");
    clap_complete::generate(shell, &mut cmd, bin_name, &mut buf);
    if buf == current.as_bytes() {
        return None;
    }
    Some(String::from_utf8(buf).expect("completion script should be UTF-8"))
}
