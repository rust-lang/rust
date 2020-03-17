//! Command like parsing for rust-analyzer.
//!
//! If run started args, we run the LSP server loop. With a subcommand, we do a
//! one-time batch processing.

use anyhow::{bail, Result};
use pico_args::Arguments;
use rust_analyzer::cli::{BenchWhat, Position, Verbosity};

use std::{fmt::Write, path::PathBuf};

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
    Stats {
        randomize: bool,
        memory_usage: bool,
        only: Option<String>,
        with_deps: bool,
        path: PathBuf,
        load_output_dirs: bool,
    },
    Bench {
        path: PathBuf,
        what: BenchWhat,
        load_output_dirs: bool,
    },
    RunServer,
    Version,
}

impl Args {
    pub(crate) fn parse() -> Result<Result<Args, HelpPrinted>> {
        let mut matches = Arguments::from_env();

        if matches.contains("--version") {
            matches.finish().or_else(handle_extra_flags)?;
            return Ok(Ok(Args { verbosity: Verbosity::Normal, command: Command::Version }));
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

        let subcommand = match matches.subcommand()? {
            Some(it) => it,
            None => {
                matches.finish().or_else(handle_extra_flags)?;
                return Ok(Ok(Args { verbosity, command: Command::RunServer }));
            }
        };
        let command = match subcommand.as_str() {
            "parse" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
ra-cli-parse

USAGE:
    rust-analyzer parse [FLAGS]

FLAGS:
    -h, --help       Prints help inforamtion
        --no-dump"
                    );
                    return Ok(Err(HelpPrinted));
                }

                let no_dump = matches.contains("--no-dump");
                matches.finish().or_else(handle_extra_flags)?;
                Command::Parse { no_dump }
            }
            "symbols" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
ra-cli-symbols

USAGE:
    rust-analyzer highlight [FLAGS]

FLAGS:
    -h, --help    Prints help inforamtion"
                    );
                    return Ok(Err(HelpPrinted));
                }

                matches.finish().or_else(handle_extra_flags)?;

                Command::Symbols
            }
            "highlight" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
ra-cli-highlight

USAGE:
    rust-analyzer highlight [FLAGS]

FLAGS:
    -h, --help       Prints help information
    -r, --rainbow"
                    );
                    return Ok(Err(HelpPrinted));
                }

                let rainbow = matches.contains(["-r", "--rainbow"]);
                matches.finish().or_else(handle_extra_flags)?;
                Command::Highlight { rainbow }
            }
            "analysis-stats" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
ra-cli-analysis-stats

USAGE:
    rust-analyzer analysis-stats [FLAGS] [OPTIONS] [PATH]

FLAGS:
    -h, --help              Prints help information
        --memory-usage
        --load-output-dirs  Load OUT_DIR values by running `cargo check` before analysis
    -v, --verbose
    -q, --quiet

OPTIONS:
    -o <ONLY>

ARGS:
    <PATH>"
                    );
                    return Ok(Err(HelpPrinted));
                }

                let randomize = matches.contains("--randomize");
                let memory_usage = matches.contains("--memory-usage");
                let only: Option<String> = matches.opt_value_from_str(["-o", "--only"])?;
                let with_deps: bool = matches.contains("--with-deps");
                let load_output_dirs = matches.contains("--load-output-dirs");
                let path = {
                    let mut trailing = matches.free()?;
                    if trailing.len() != 1 {
                        bail!("Invalid flags");
                    }
                    trailing.pop().unwrap().into()
                };

                Command::Stats { randomize, memory_usage, only, with_deps, path, load_output_dirs }
            }
            "analysis-bench" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
rust-analyzer-analysis-bench

USAGE:
    rust-analyzer analysis-bench [FLAGS] [OPTIONS]

FLAGS:
    -h, --help          Prints help information
    --load-output-dirs  Load OUT_DIR values by running `cargo check` before analysis
    -v, --verbose

OPTIONS:
    --project <PATH>                 Path to directory with Cargo.toml
    --complete <PATH:LINE:COLUMN>    Compute completions at this location
    --goto-def <PATH:LINE:COLUMN>    Compute goto definition at this location
    --highlight <PATH>               Hightlight this file

ARGS:
    <PATH>    Project to analyse"
                    );
                    return Ok(Err(HelpPrinted));
                }

                let path: PathBuf = matches.opt_value_from_str("--project")?.unwrap_or_default();
                let highlight_path: Option<String> = matches.opt_value_from_str("--highlight")?;
                let complete_path: Option<Position> = matches.opt_value_from_str("--complete")?;
                let goto_def_path: Option<Position> = matches.opt_value_from_str("--goto-def")?;
                let what = match (highlight_path, complete_path, goto_def_path) {
                    (Some(path), None, None) => BenchWhat::Highlight { path: path.into() },
                    (None, Some(position), None) => BenchWhat::Complete(position),
                    (None, None, Some(position)) => BenchWhat::GotoDef(position),
                    _ => panic!(
                        "exactly one of  `--highlight`, `--complete` or `--goto-def` must be set"
                    ),
                };
                let load_output_dirs = matches.contains("--load-output-dirs");
                Command::Bench { path, what, load_output_dirs }
            }
            _ => {
                eprintln!(
                    "\
ra-cli

USAGE:
    rust-analyzer <SUBCOMMAND>

FLAGS:
    -h, --help        Prints help information

SUBCOMMANDS:
    analysis-bench
    analysis-stats
    highlight
    parse
    symbols"
                );
                return Ok(Err(HelpPrinted));
            }
        };
        Ok(Ok(Args { verbosity, command }))
    }
}

pub(crate) struct HelpPrinted;

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
