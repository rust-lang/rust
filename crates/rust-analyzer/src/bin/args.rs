//! Command like parsing for rust-analyzer.
//!
//! If run started args, we run the LSP server loop. With a subcommand, we do a
//! one-time batch processing.

use std::{env, fmt::Write, path::PathBuf};

use anyhow::{bail, Result};
use pico_args::Arguments;
use rust_analyzer::cli::{AnalysisStatsCmd, BenchCmd, BenchWhat, Position, Verbosity};
use ssr::{SsrPattern, SsrRule};
use vfs::AbsPathBuf;

pub(crate) struct Args {
    pub(crate) verbosity: Verbosity,
    pub(crate) command: Command,
}

pub(crate) enum Command {
    Parse {
        no_dump: bool,
    },
    Symbols,
    Highlight {
        rainbow: bool,
    },
    AnalysisStats(AnalysisStatsCmd),
    Bench(BenchCmd),
    Diagnostics {
        path: PathBuf,
        load_output_dirs: bool,
        with_proc_macro: bool,
        /// Include files which are not modules. In rust-analyzer
        /// this would include the parser test files.
        all: bool,
    },
    Ssr {
        rules: Vec<SsrRule>,
    },
    StructuredSearch {
        debug_snippet: Option<String>,
        patterns: Vec<SsrPattern>,
    },
    ProcMacro,
    RunServer,
    Version,
    Help,
}

impl Args {
    pub(crate) fn parse() -> Result<Args> {
        let mut matches = Arguments::from_env();

        if matches.contains("--version") {
            matches.finish().or_else(handle_extra_flags)?;
            return Ok(Args { verbosity: Verbosity::Normal, command: Command::Version });
        }

        let verbosity = match (
            matches.contains(["-vv", "--spammy"]),
            matches.contains(["-v", "--verbose"]),
            matches.contains(["-q", "--quiet"]),
        ) {
            (true, _, true) => bail!("Invalid flags: -q conflicts with -vv"),
            (true, _, false) => Verbosity::Spammy,
            (false, false, false) => Verbosity::Normal,
            (false, false, true) => Verbosity::Quiet,
            (false, true, false) => Verbosity::Verbose,
            (false, true, true) => bail!("Invalid flags: -q conflicts with -v"),
        };

        let help = Ok(Args { verbosity, command: Command::Help });
        let subcommand = match matches.subcommand()? {
            Some(it) => it,
            None => {
                if matches.contains(["-h", "--help"]) {
                    print_subcommands();
                    return help;
                }
                matches.finish().or_else(handle_extra_flags)?;
                return Ok(Args { verbosity, command: Command::RunServer });
            }
        };
        let command = match subcommand.as_str() {
            "parse" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer parse

USAGE:
    rust-analyzer parse [FLAGS]

FLAGS:
    -h, --help       Prints help information
        --no-dump"
                    );
                    return help;
                }

                let no_dump = matches.contains("--no-dump");
                matches.finish().or_else(handle_extra_flags)?;
                Command::Parse { no_dump }
            }
            "symbols" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer symbols

USAGE:
    rust-analyzer highlight [FLAGS]

FLAGS:
    -h, --help    Prints help inforamtion"
                    );
                    return help;
                }

                matches.finish().or_else(handle_extra_flags)?;

                Command::Symbols
            }
            "highlight" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer highlight

USAGE:
    rust-analyzer highlight [FLAGS]

FLAGS:
    -h, --help       Prints help information
    -r, --rainbow"
                    );
                    return help;
                }

                let rainbow = matches.contains(["-r", "--rainbow"]);
                matches.finish().or_else(handle_extra_flags)?;
                Command::Highlight { rainbow }
            }
            "analysis-stats" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer analysis-stats

USAGE:
    rust-analyzer analysis-stats [FLAGS] [OPTIONS] [PATH]

FLAGS:
    -o, --only              Only analyze items matching this path
    -h, --help              Prints help information
        --memory-usage      Collect memory usage statistics
        --randomize         Randomize order in which crates, modules, and items are processed
        --parallel          Run type inference in parallel
        --load-output-dirs  Load OUT_DIR values by running `cargo check` before analysis
        --with-proc-macro   Use ra-proc-macro-srv for proc-macro expanding
        --with-deps         Also analyze all dependencies
    -v, --verbose
    -q, --quiet

OPTIONS:
    -o <ONLY>

ARGS:
    <PATH>"
                    );
                    return help;
                }

                let randomize = matches.contains("--randomize");
                let parallel = matches.contains("--parallel");
                let memory_usage = matches.contains("--memory-usage");
                let only: Option<String> = matches.opt_value_from_str(["-o", "--only"])?;
                let with_deps: bool = matches.contains("--with-deps");
                let load_output_dirs = matches.contains("--load-output-dirs");
                let with_proc_macro = matches.contains("--with-proc-macro");
                let path = {
                    let mut trailing = matches.free()?;
                    if trailing.len() != 1 {
                        bail!("Invalid flags");
                    }
                    trailing.pop().unwrap().into()
                };

                Command::AnalysisStats(AnalysisStatsCmd {
                    randomize,
                    parallel,
                    memory_usage,
                    only,
                    with_deps,
                    path,
                    load_output_dirs,
                    with_proc_macro,
                })
            }
            "analysis-bench" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer analysis-bench

USAGE:
    rust-analyzer analysis-bench [FLAGS] [OPTIONS]

FLAGS:
    -h, --help          Prints help information
    --memory-usage      Collect memory usage statistics
    --load-output-dirs  Load OUT_DIR values by running `cargo check` before analysis
    --with-proc-macro   Use ra-proc-macro-srv for proc-macro expanding
    -v, --verbose

OPTIONS:
    --project <PATH>                 Path to directory with Cargo.toml
    --complete <PATH:LINE:COLUMN>    Compute completions at this location
    --goto-def <PATH:LINE:COLUMN>    Compute goto definition at this location
    --highlight <PATH>               Hightlight this file

ARGS:
    <PATH>    Project to analyse"
                    );
                    return help;
                }

                let path: PathBuf = matches.opt_value_from_str("--project")?.unwrap_or_default();
                let highlight_path: Option<String> = matches.opt_value_from_str("--highlight")?;
                let complete_path: Option<Position> = matches.opt_value_from_str("--complete")?;
                let goto_def_path: Option<Position> = matches.opt_value_from_str("--goto-def")?;
                let what = match (highlight_path, complete_path, goto_def_path) {
                    (Some(path), None, None) => {
                        let path = env::current_dir().unwrap().join(path);
                        BenchWhat::Highlight { path: AbsPathBuf::assert(path) }
                    }
                    (None, Some(position), None) => BenchWhat::Complete(position),
                    (None, None, Some(position)) => BenchWhat::GotoDef(position),
                    _ => panic!(
                        "exactly one of  `--highlight`, `--complete` or `--goto-def` must be set"
                    ),
                };
                let memory_usage = matches.contains("--memory-usage");
                let load_output_dirs = matches.contains("--load-output-dirs");
                let with_proc_macro = matches.contains("--with-proc-macro");
                Command::Bench(BenchCmd {
                    memory_usage,
                    path,
                    what,
                    load_output_dirs,
                    with_proc_macro,
                })
            }
            "diagnostics" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer diagnostics

USAGE:
    rust-analyzer diagnostics [FLAGS] [PATH]

FLAGS:
    -h, --help              Prints help information
        --load-output-dirs  Load OUT_DIR values by running `cargo check` before analysis
        --all               Include all files rather than only modules

ARGS:
    <PATH>"
                    );
                    return help;
                }

                let load_output_dirs = matches.contains("--load-output-dirs");
                let with_proc_macro = matches.contains("--with-proc-macro");
                let all = matches.contains("--all");
                let path = {
                    let mut trailing = matches.free()?;
                    if trailing.len() != 1 {
                        bail!("Invalid flags");
                    }
                    trailing.pop().unwrap().into()
                };

                Command::Diagnostics { path, load_output_dirs, with_proc_macro, all }
            }
            "proc-macro" => Command::ProcMacro,
            "ssr" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer ssr

USAGE:
    rust-analyzer ssr [FLAGS] [RULE...]

EXAMPLE:
    rust-analyzer ssr '$a.foo($b) ==> bar($a, $b)'

FLAGS:
    --debug <snippet>   Prints debug information for any nodes with source exactly equal to <snippet>
    -h, --help          Prints help information

ARGS:
    <RULE>              A structured search replace rule"
                    );
                    return help;
                }
                let mut rules = Vec::new();
                while let Some(rule) = matches.free_from_str()? {
                    rules.push(rule);
                }
                Command::Ssr { rules }
            }
            "search" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer search

USAGE:
    rust-analyzer search [FLAGS] [PATTERN...]

EXAMPLE:
    rust-analyzer search '$a.foo($b)'

FLAGS:
    --debug <snippet>   Prints debug information for any nodes with source exactly equal to <snippet>
    -h, --help          Prints help information

ARGS:
    <PATTERN>           A structured search pattern"
                    );
                    return help;
                }
                let debug_snippet = matches.opt_value_from_str("--debug")?;
                let mut patterns = Vec::new();
                while let Some(rule) = matches.free_from_str()? {
                    patterns.push(rule);
                }
                Command::StructuredSearch { patterns, debug_snippet }
            }
            _ => {
                print_subcommands();
                return help;
            }
        };
        Ok(Args { verbosity, command })
    }
}

fn print_subcommands() {
    eprintln!(
        "\
rust-analyzer

USAGE:
    rust-analyzer <SUBCOMMAND>

FLAGS:
    -h, --help        Prints help information

SUBCOMMANDS:
    analysis-bench
    analysis-stats
    highlight
    diagnostics
    proc-macro
    parse
    search
    ssr
    symbols"
    )
}

fn handle_extra_flags(e: pico_args::Error) -> Result<()> {
    if let pico_args::Error::UnusedArgsLeft(flags) = e {
        let mut invalid_flags = String::new();
        for flag in flags {
            write!(&mut invalid_flags, "{}, ", flag)?;
        }
        let (invalid_flags, _) = invalid_flags.split_at(invalid_flags.len() - 2);
        bail!("Invalid flags: {}", invalid_flags);
    } else {
        bail!(e);
    }
}
