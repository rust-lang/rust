//! FIXME: write short doc here

mod load_cargo;
mod analysis_stats;
mod analysis_bench;
mod progress_report;

use std::{error::Error, fmt::Write, io::Read, path::PathBuf};

use pico_args::Arguments;
use ra_ide::{file_structure, Analysis};
use ra_prof::profile;
use ra_syntax::{AstNode, SourceFile};

type Result<T, E = Box<dyn Error + Send + Sync>> = std::result::Result<T, E>;

fn main() -> Result<()> {
    env_logger::try_init()?;

    let command = match Command::from_env_args()? {
        Ok(it) => it,
        Err(HelpPrinted) => return Ok(()),
    };
    match command {
        Command::Parse { no_dump } => {
            let _p = profile("parsing");
            let file = file()?;
            if !no_dump {
                println!("{:#?}", file.syntax());
            }
            std::mem::forget(file);
        }
        Command::Symbols => {
            let file = file()?;
            for s in file_structure(&file) {
                println!("{:?}", s);
            }
        }
        Command::Highlight { rainbow } => {
            let (analysis, file_id) = Analysis::from_single_file(read_stdin()?);
            let html = analysis.highlight_as_html(file_id, rainbow).unwrap();
            println!("{}", html);
        }
        Command::Stats { verbosity, randomize, memory_usage, only, with_deps, path } => {
            analysis_stats::run(
                verbosity,
                memory_usage,
                path.as_ref(),
                only.as_ref().map(String::as_ref),
                with_deps,
                randomize,
            )?;
        }
        Command::Bench { verbosity, path, op } => {
            analysis_bench::run(verbosity, path.as_ref(), op)?;
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
pub enum Verbosity {
    Spammy,
    Verbose,
    Normal,
    Quiet,
}

impl Verbosity {
    fn is_verbose(self) -> bool {
        match self {
            Verbosity::Verbose | Verbosity::Spammy => true,
            _ => false,
        }
    }
    fn is_spammy(self) -> bool {
        match self {
            Verbosity::Spammy => true,
            _ => false,
        }
    }
}

enum Command {
    Parse {
        no_dump: bool,
    },
    Symbols,
    Highlight {
        rainbow: bool,
    },
    Stats {
        verbosity: Verbosity,
        randomize: bool,
        memory_usage: bool,
        only: Option<String>,
        with_deps: bool,
        path: PathBuf,
    },
    Bench {
        verbosity: Verbosity,
        path: PathBuf,
        op: analysis_bench::Op,
    },
}

struct HelpPrinted;

impl Command {
    fn from_env_args() -> Result<Result<Command, HelpPrinted>> {
        let mut matches = Arguments::from_env();
        let subcommand = matches.subcommand()?.unwrap_or_default();

        let verbosity = match (
            matches.contains(["-vv", "--spammy"]),
            matches.contains(["-v", "--verbose"]),
            matches.contains(["-q", "--quiet"]),
        ) {
            (true, _, true) => Err("Invalid flags: -q conflicts with -vv")?,
            (true, _, false) => Verbosity::Spammy,
            (false, false, false) => Verbosity::Normal,
            (false, false, true) => Verbosity::Quiet,
            (false, true, false) => Verbosity::Verbose,
            (false, true, true) => Err("Invalid flags: -q conflicts with -v")?,
        };

        let command = match subcommand.as_str() {
            "parse" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
ra-cli-parse

USAGE:
    ra_cli parse [FLAGS]

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
    ra_cli highlight [FLAGS]

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
    ra_cli highlight [FLAGS]

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
    ra_cli analysis-stats [FLAGS] [OPTIONS] [PATH]

FLAGS:
    -h, --help            Prints help information
        --memory-usage
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
                let path = {
                    let mut trailing = matches.free()?;
                    if trailing.len() != 1 {
                        Err("Invalid flags")?;
                    }
                    trailing.pop().unwrap().into()
                };

                Command::Stats { verbosity, randomize, memory_usage, only, with_deps, path }
            }
            "analysis-bench" => {
                if matches.contains(["-h", "--help"]) {
                    eprintln!(
                        "\
ra_cli-analysis-bench

USAGE:
    ra_cli analysis-bench [FLAGS] [OPTIONS] [PATH]

FLAGS:
    -h, --help        Prints help information
    -v, --verbose

OPTIONS:
    --complete <PATH:LINE:COLUMN>    Compute completions at this location
    --highlight <PATH>               Hightlight this file

ARGS:
    <PATH>    Project to analyse"
                    );
                    return Ok(Err(HelpPrinted));
                }

                let path: PathBuf = matches.opt_value_from_str("--path")?.unwrap_or_default();
                let highlight_path: Option<String> = matches.opt_value_from_str("--highlight")?;
                let complete_path: Option<String> = matches.opt_value_from_str("--complete")?;
                let goto_def_path: Option<String> = matches.opt_value_from_str("--goto-def")?;
                let op = match (highlight_path, complete_path, goto_def_path) {
                    (Some(path), None, None) => analysis_bench::Op::Highlight { path: path.into() },
                    (None, Some(position), None) => analysis_bench::Op::Complete(position.parse()?),
                    (None, None, Some(position)) => analysis_bench::Op::GotoDef(position.parse()?),
                    _ => panic!(
                        "exactly one of  `--highlight`, `--complete` or `--goto-def` must be set"
                    ),
                };
                Command::Bench { verbosity, path, op }
            }
            _ => {
                eprintln!(
                    "\
ra-cli

USAGE:
    ra_cli <SUBCOMMAND>

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
        Ok(Ok(command))
    }
}

fn handle_extra_flags(e: pico_args::Error) -> Result<()> {
    if let pico_args::Error::UnusedArgsLeft(flags) = e {
        let mut invalid_flags = String::new();
        for flag in flags {
            write!(&mut invalid_flags, "{}, ", flag)?;
        }
        let (invalid_flags, _) = invalid_flags.split_at(invalid_flags.len() - 2);
        Err(format!("Invalid flags: {}", invalid_flags).into())
    } else {
        Err(e.to_string().into())
    }
}

fn file() -> Result<SourceFile> {
    let text = read_stdin()?;
    Ok(SourceFile::parse(&text).tree())
}

fn read_stdin() -> Result<String> {
    let mut buff = String::new();
    std::io::stdin().read_to_string(&mut buff)?;
    Ok(buff)
}
