//! Grammar for the command-line arguments.
#![allow(unreachable_pub)]
use std::{path::PathBuf, str::FromStr};

use ide_ssr::{SsrPattern, SsrRule};

use crate::cli::Verbosity;

xflags::xflags! {
    src "./src/cli/flags.rs"

    /// LSP server for the Rust programming language.
    ///
    /// Subcommands and their flags do not provide any stability guarantees and may be removed or
    /// changed without notice. Top-level flags that are not marked as [Unstable] provide
    /// backwards-compatibility and may be relied on.
    cmd rust-analyzer {
        /// Verbosity level, can be repeated multiple times.
        repeated -v, --verbose
        /// Verbosity level.
        optional -q, --quiet

        /// Log to the specified file instead of stderr.
        optional --log-file path: PathBuf
        /// Flush log records to the file immediately.
        optional --no-log-buffering

        /// [Unstable] Wait until a debugger is attached to (requires debug build).
        optional --wait-dbg

        default cmd lsp-server {
            /// Print version.
            optional -V, --version

            /// Dump a LSP config JSON schema.
            optional --print-config-schema
        }

        /// Parse stdin.
        cmd parse {
            /// Suppress printing.
            optional --no-dump
        }

        /// Parse stdin and print the list of symbols.
        cmd symbols {}

        /// Highlight stdin as html.
        cmd highlight {
            /// Enable rainbow highlighting of identifiers.
            optional --rainbow
        }

        /// Batch typecheck project and print summary statistics
        cmd analysis-stats {
            /// Directory with Cargo.toml or rust-project.json.
            required path: PathBuf

            optional --output format: OutputFormat

            /// Randomize order in which crates, modules, and items are processed.
            optional --randomize
            /// Run type inference in parallel.
            optional --parallel

            /// Only analyze items matching this path.
            optional -o, --only path: String
            /// Also analyze all dependencies.
            optional --with-deps
            /// Don't load sysroot crates (`std`, `core` & friends).
            optional --no-sysroot
            /// Don't set #[cfg(test)].
            optional --no-test

            /// Don't run build scripts or load `OUT_DIR` values by running `cargo check` before analysis.
            optional --disable-build-scripts
            /// Don't expand proc macros.
            optional --disable-proc-macros
            /// Run the proc-macro-srv binary at the specified path.
            optional --proc-macro-srv path: PathBuf
            /// Skip body lowering.
            optional --skip-lowering
            /// Skip type inference.
            optional --skip-inference
            /// Skip lowering to mir
            optional --skip-mir-stats
            /// Skip data layout calculation
            optional --skip-data-layout
            /// Skip const evaluation
            optional --skip-const-eval
            /// Runs several IDE features after analysis, including semantics highlighting, diagnostics
            /// and annotations. This is useful for benchmarking the memory usage on a project that has
            /// been worked on for a bit in a longer running session.
            optional --run-all-ide-things
            /// Run term search on all the tail expressions (of functions, block, if statements etc.)
            optional --run-term-search
            /// Validate term search by running `cargo check` on every response.
            /// Note that this also temporarily modifies the files on disk, use with caution!
            optional --validate-term-search
        }

        /// Run unit tests of the project using mir interpreter
        cmd run-tests {
            /// Directory with Cargo.toml or rust-project.json.
            required path: PathBuf
        }

        /// Run unit tests of the project using mir interpreter
        cmd rustc-tests {
            /// Directory with Cargo.toml.
            required rustc_repo: PathBuf

            /// Only run tests with filter as substring
            optional --filter path: String
        }

        cmd diagnostics {
            /// Directory with Cargo.toml or rust-project.json.
            required path: PathBuf

            /// Don't run build scripts or load `OUT_DIR` values by running `cargo check` before analysis.
            optional --disable-build-scripts
            /// Don't expand proc macros.
            optional --disable-proc-macros
            /// Run the proc-macro-srv binary at the specified path.
            optional --proc-macro-srv path: PathBuf
        }

        /// Report unresolved references
        cmd unresolved-references {
            /// Directory with Cargo.toml or rust-project.json.
            required path: PathBuf

            /// Don't run build scripts or load `OUT_DIR` values by running `cargo check` before analysis.
            optional --disable-build-scripts
            /// Don't expand proc macros.
            optional --disable-proc-macros
            /// Run the proc-macro-srv binary at the specified path.
            optional --proc-macro-srv path: PathBuf
        }

        /// Prime caches, as rust-analyzer does typically at startup in interactive sessions.
        cmd prime-caches {
            /// Directory with Cargo.toml or rust-project.json.
            required path: PathBuf

            /// Don't run build scripts or load `OUT_DIR` values by running `cargo check` before analysis.
            optional --disable-build-scripts
            /// Don't expand proc macros.
            optional --disable-proc-macros
            /// Run the proc-macro-srv binary at the specified path.
            optional --proc-macro-srv path: PathBuf
            /// The number of threads to use. Defaults to the number of physical cores.
            optional --num-threads num_threads: usize
        }

        cmd ssr {
            /// A structured search replace rule (`$a.foo($b) ==>> bar($a, $b)`)
            repeated rule: SsrRule
        }

        cmd search {
            /// A structured search replace pattern (`$a.foo($b)`)
            repeated pattern: SsrPattern
            /// Prints debug information for any nodes with source exactly equal to snippet.
            optional --debug snippet: String
        }

        cmd lsif {
            required path: PathBuf

            /// Exclude code from vendored libraries from the resulting index.
            optional --exclude-vendored-libraries
        }

        cmd scip {
            required path: PathBuf

            /// The output path where the SCIP file will be written to. Defaults to `index.scip`.
            optional --output path: PathBuf

            /// A path to an json configuration file that can be used to customize cargo behavior.
            optional --config-path config_path: PathBuf

            /// Exclude code from vendored libraries from the resulting index.
            optional --exclude-vendored-libraries
        }
    }
}

// generated start
// The following code is generated by `xflags` macro.
// Run `env UPDATE_XFLAGS=1 cargo build` to regenerate.
#[derive(Debug)]
pub struct RustAnalyzer {
    pub verbose: u32,
    pub quiet: bool,
    pub log_file: Option<PathBuf>,
    pub no_log_buffering: bool,
    pub wait_dbg: bool,
    pub subcommand: RustAnalyzerCmd,
}

#[derive(Debug)]
pub enum RustAnalyzerCmd {
    LspServer(LspServer),
    Parse(Parse),
    Symbols(Symbols),
    Highlight(Highlight),
    AnalysisStats(AnalysisStats),
    RunTests(RunTests),
    RustcTests(RustcTests),
    Diagnostics(Diagnostics),
    UnresolvedReferences(UnresolvedReferences),
    PrimeCaches(PrimeCaches),
    Ssr(Ssr),
    Search(Search),
    Lsif(Lsif),
    Scip(Scip),
}

#[derive(Debug)]
pub struct LspServer {
    pub version: bool,
    pub print_config_schema: bool,
}

#[derive(Debug)]
pub struct Parse {
    pub no_dump: bool,
}

#[derive(Debug)]
pub struct Symbols;

#[derive(Debug)]
pub struct Highlight {
    pub rainbow: bool,
}

#[derive(Debug)]
pub struct AnalysisStats {
    pub path: PathBuf,

    pub output: Option<OutputFormat>,
    pub randomize: bool,
    pub parallel: bool,
    pub only: Option<String>,
    pub with_deps: bool,
    pub no_sysroot: bool,
    pub no_test: bool,
    pub disable_build_scripts: bool,
    pub disable_proc_macros: bool,
    pub proc_macro_srv: Option<PathBuf>,
    pub skip_lowering: bool,
    pub skip_inference: bool,
    pub skip_mir_stats: bool,
    pub skip_data_layout: bool,
    pub skip_const_eval: bool,
    pub run_all_ide_things: bool,
    pub run_term_search: bool,
    pub validate_term_search: bool,
}

#[derive(Debug)]
pub struct RunTests {
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct RustcTests {
    pub rustc_repo: PathBuf,

    pub filter: Option<String>,
}

#[derive(Debug)]
pub struct Diagnostics {
    pub path: PathBuf,

    pub disable_build_scripts: bool,
    pub disable_proc_macros: bool,
    pub proc_macro_srv: Option<PathBuf>,
}

#[derive(Debug)]
pub struct UnresolvedReferences {
    pub path: PathBuf,

    pub disable_build_scripts: bool,
    pub disable_proc_macros: bool,
    pub proc_macro_srv: Option<PathBuf>,
}

#[derive(Debug)]
pub struct PrimeCaches {
    pub path: PathBuf,

    pub disable_build_scripts: bool,
    pub disable_proc_macros: bool,
    pub proc_macro_srv: Option<PathBuf>,
    pub num_threads: Option<usize>,
}

#[derive(Debug)]
pub struct Ssr {
    pub rule: Vec<SsrRule>,
}

#[derive(Debug)]
pub struct Search {
    pub pattern: Vec<SsrPattern>,

    pub debug: Option<String>,
}

#[derive(Debug)]
pub struct Lsif {
    pub path: PathBuf,

    pub exclude_vendored_libraries: bool,
}

#[derive(Debug)]
pub struct Scip {
    pub path: PathBuf,

    pub output: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub exclude_vendored_libraries: bool,
}

impl RustAnalyzer {
    #[allow(dead_code)]
    pub fn from_env_or_exit() -> Self {
        Self::from_env_or_exit_()
    }

    #[allow(dead_code)]
    pub fn from_env() -> xflags::Result<Self> {
        Self::from_env_()
    }

    #[allow(dead_code)]
    pub fn from_vec(args: Vec<std::ffi::OsString>) -> xflags::Result<Self> {
        Self::from_vec_(args)
    }
}
// generated end

#[derive(Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Csv,
}

impl RustAnalyzer {
    pub fn verbosity(&self) -> Verbosity {
        if self.quiet {
            return Verbosity::Quiet;
        }
        match self.verbose {
            0 => Verbosity::Normal,
            1 => Verbosity::Verbose,
            _ => Verbosity::Spammy,
        }
    }
}

impl FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "csv" => Ok(Self::Csv),
            _ => Err(format!("unknown output format `{s}`")),
        }
    }
}
