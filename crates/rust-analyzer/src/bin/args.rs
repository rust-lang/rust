//! Command like parsing for rust-analyzer.
//!
//! If run started args, we run the LSP server loop. With a subcommand, we do a
//! one-time batch processing.

use std::{env, path::PathBuf};

use anyhow::{bail, format_err, Result};
use pico_args::Arguments;
use rust_analyzer::cli::{AnalysisStatsCmd, BenchCmd, BenchWhat, Position, Verbosity};
use ssr::{SsrPattern, SsrRule};
use vfs::AbsPathBuf;

pub(crate) struct Args {
    pub(crate) verbosity: Verbosity,
    pub(crate) log_file: Option<PathBuf>,
    pub(crate) no_buffering: bool,
    pub(crate) command: Command,
    #[allow(unused)]
    pub(crate) wait_dbg: bool,
}

pub(crate) enum Command {
    Parse { no_dump: bool },
    Symbols,
    Highlight { rainbow: bool },
    AnalysisStats(AnalysisStatsCmd),
    Bench(BenchCmd),
    Diagnostics { path: PathBuf, load_output_dirs: bool, with_proc_macro: bool },
    Ssr { rules: Vec<SsrRule> },
    StructuredSearch { debug_snippet: Option<String>, patterns: Vec<SsrPattern> },
    ProcMacro,
    RunServer,
    PrintConfigSchema,
    Version,
    Help,
}

const HELP: &str = "\
rust-analyzer

USAGE:
    rust-analyzer [FLAGS] [COMMAND] [COMMAND_OPTIONS]

FLAGS:
    --version         Print version
    -h, --help        Print this help

    -v,  --verbose
    -vv, --spammy
    -q,  --quiet      Set verbosity

    --print-config-schema
                      Dump a LSP config JSON schema
    --log-file <PATH> Log to the specified file instead of stderr
    --no-log-buffering
                      Flush log records to the file immediately

    --wait-dbg        Wait until a debugger is attached to.
                      The flag is valid for debug builds only

ENVIRONMENTAL VARIABLES:
    RA_LOG            Set log filter in env_logger format
    RA_PROFILE        Enable hierarchical profiler
    RA_WAIT_DBG       If set acts like a --wait-dbg flag

COMMANDS:

not specified         Launch LSP server

parse < main.rs       Parse tree
    --no-dump         Suppress printing

symbols < main.rs     Parse input an print the list of symbols

highlight < main.rs   Highlight input as html
    --rainbow         Enable rainbow highlighting of identifiers

analysis-stats <PATH> Batch typecheck project and print summary statistics
    <PATH>            Directory with Cargo.toml
    --randomize       Randomize order in which crates, modules, and items are processed
    --parallel        Run type inference in parallel
    --memory-usage    Collect memory usage statistics
    -o, --only <PATH> Only analyze items matching this path
    --with-deps       Also analyze all dependencies
    --load-output-dirs
                      Load OUT_DIR values by running `cargo check` before analysis
    --with-proc-macro Use proc-macro-srv for proc-macro expanding

analysis-bench <PATH> Benchmark specific analysis operation
    <PATH>            Directory with Cargo.toml
    --highlight <PATH>
                      Compute syntax highlighting for this file
    --complete <PATH:LINE:COLUMN>
                      Compute completions at this location
    --goto-def <PATH:LINE:COLUMN>
                      Compute goto definition at this location
    --memory-usage    Collect memory usage statistics
    --load-output-dirs
                      Load OUT_DIR values by running `cargo check` before analysis
    --with-proc-macro Use proc-macro-srv for proc-macro expanding

diagnostics <PATH>
    <PATH>            Directory with Cargo.toml
    --load-output-dirs
                      Load OUT_DIR values by running `cargo check` before analysis
    --with-proc-macro Use proc-macro-srv for proc-macro expanding

ssr [RULE...]
    <RULE>            A structured search replace rule (`$a.foo($b) ==> bar($a, $b)`)

search [PATTERN..]
    <PATTERN>         A structured search replace pattern (`$a.foo($b)`)
    --debug <snippet> Prints debug information for any nodes with source exactly
                      equal to <snippet>
";

impl Args {
    pub(crate) fn parse() -> Result<Args> {
        let mut matches = Arguments::from_env();

        if matches.contains("--version") {
            finish_args(matches)?;
            return Ok(Args {
                verbosity: Verbosity::Normal,
                log_file: None,
                command: Command::Version,
                no_buffering: false,
                wait_dbg: false,
            });
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
        let log_file = matches.opt_value_from_str("--log-file")?;
        let no_buffering = matches.contains("--no-log-buffering");
        let wait_dbg = matches.contains("--wait-dbg");

        if matches.contains(["-h", "--help"]) {
            eprintln!("{}", HELP);
            return Ok(Args {
                verbosity,
                log_file: None,
                command: Command::Help,
                no_buffering,
                wait_dbg,
            });
        }

        if matches.contains("--print-config-schema") {
            return Ok(Args {
                verbosity,
                log_file,
                command: Command::PrintConfigSchema,
                no_buffering,
                wait_dbg,
            });
        }

        let subcommand = match matches.subcommand()? {
            Some(it) => it,
            None => {
                finish_args(matches)?;
                return Ok(Args {
                    verbosity,
                    log_file,
                    command: Command::RunServer,
                    no_buffering,
                    wait_dbg,
                });
            }
        };
        let command = match subcommand.as_str() {
            "parse" => Command::Parse { no_dump: matches.contains("--no-dump") },
            "symbols" => Command::Symbols,
            "highlight" => Command::Highlight { rainbow: matches.contains("--rainbow") },
            "analysis-stats" => Command::AnalysisStats(AnalysisStatsCmd {
                randomize: matches.contains("--randomize"),
                parallel: matches.contains("--parallel"),
                memory_usage: matches.contains("--memory-usage"),
                only: matches.opt_value_from_str(["-o", "--only"])?,
                with_deps: matches.contains("--with-deps"),
                load_output_dirs: matches.contains("--load-output-dirs"),
                with_proc_macro: matches.contains("--with-proc-macro"),
                path: matches
                    .opt_free_from_str()?
                    .ok_or_else(|| format_err!("expected positional argument"))?,
            }),
            "analysis-bench" => Command::Bench(BenchCmd {
                what: {
                    let highlight_path: Option<String> =
                        matches.opt_value_from_str("--highlight")?;
                    let complete_path: Option<Position> =
                        matches.opt_value_from_str("--complete")?;
                    let goto_def_path: Option<Position> =
                        matches.opt_value_from_str("--goto-def")?;
                    match (highlight_path, complete_path, goto_def_path) {
                            (Some(path), None, None) => {
                                let path = env::current_dir().unwrap().join(path);
                                BenchWhat::Highlight { path: AbsPathBuf::assert(path) }
                            }
                            (None, Some(position), None) => BenchWhat::Complete(position),
                            (None, None, Some(position)) => BenchWhat::GotoDef(position),
                            _ => panic!(
                                "exactly one of  `--highlight`, `--complete` or `--goto-def` must be set"
                            ),
                        }
                },
                memory_usage: matches.contains("--memory-usage"),
                load_output_dirs: matches.contains("--load-output-dirs"),
                with_proc_macro: matches.contains("--with-proc-macro"),
                path: matches
                    .opt_free_from_str()?
                    .ok_or_else(|| format_err!("expected positional argument"))?,
            }),
            "diagnostics" => Command::Diagnostics {
                load_output_dirs: matches.contains("--load-output-dirs"),
                with_proc_macro: matches.contains("--with-proc-macro"),
                path: matches
                    .opt_free_from_str()?
                    .ok_or_else(|| format_err!("expected positional argument"))?,
            },
            "proc-macro" => Command::ProcMacro,
            "ssr" => Command::Ssr {
                rules: {
                    let mut acc = Vec::new();
                    while let Some(rule) = matches.opt_free_from_str()? {
                        acc.push(rule);
                    }
                    acc
                },
            },
            "search" => Command::StructuredSearch {
                debug_snippet: matches.opt_value_from_str("--debug")?,
                patterns: {
                    let mut acc = Vec::new();
                    while let Some(rule) = matches.opt_free_from_str()? {
                        acc.push(rule);
                    }
                    acc
                },
            },
            _ => {
                eprintln!("{}", HELP);
                return Ok(Args {
                    verbosity,
                    log_file: None,
                    command: Command::Help,
                    no_buffering,
                    wait_dbg,
                });
            }
        };
        finish_args(matches)?;
        Ok(Args { verbosity, log_file, command, no_buffering, wait_dbg })
    }
}

fn finish_args(args: Arguments) -> Result<()> {
    if !args.finish().is_empty() {
        bail!("Unused arguments.");
    }
    Ok(())
}
