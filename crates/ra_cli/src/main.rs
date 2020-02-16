//! FIXME: write short doc here

mod analysis_stats;
mod analysis_bench;
mod help;
mod progress_report;

use std::{error::Error, fmt::Write, io::Read};

use pico_args::Arguments;
use ra_ide::{file_structure, Analysis};
use ra_prof::profile;
use ra_syntax::{AstNode, SourceFile};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

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

fn main() -> Result<()> {
    env_logger::try_init()?;

    let subcommand = match std::env::args_os().nth(1) {
        None => {
            eprintln!("{}", help::GLOBAL_HELP);
            return Ok(());
        }
        Some(s) => s,
    };
    let mut matches = Arguments::from_vec(std::env::args_os().skip(2).collect());

    match &*subcommand.to_string_lossy() {
        "parse" => {
            if matches.contains(["-h", "--help"]) {
                eprintln!("{}", help::PARSE_HELP);
                return Ok(());
            }
            let no_dump = matches.contains("--no-dump");
            matches.finish().or_else(handle_extra_flags)?;

            let _p = profile("parsing");
            let file = file()?;
            if !no_dump {
                println!("{:#?}", file.syntax());
            }
            std::mem::forget(file);
        }
        "symbols" => {
            if matches.contains(["-h", "--help"]) {
                eprintln!("{}", help::SYMBOLS_HELP);
                return Ok(());
            }
            matches.finish().or_else(handle_extra_flags)?;
            let file = file()?;
            for s in file_structure(&file) {
                println!("{:?}", s);
            }
        }
        "highlight" => {
            if matches.contains(["-h", "--help"]) {
                eprintln!("{}", help::HIGHLIGHT_HELP);
                return Ok(());
            }
            let rainbow_opt = matches.contains(["-r", "--rainbow"]);
            matches.finish().or_else(handle_extra_flags)?;
            let (analysis, file_id) = Analysis::from_single_file(read_stdin()?);
            let html = analysis.highlight_as_html(file_id, rainbow_opt).unwrap();
            println!("{}", html);
        }
        "analysis-stats" => {
            if matches.contains(["-h", "--help"]) {
                eprintln!("{}", help::ANALYSIS_STATS_HELP);
                return Ok(());
            }
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
            let randomize = matches.contains("--randomize");
            let memory_usage = matches.contains("--memory-usage");
            let only: Option<String> = matches.opt_value_from_str(["-o", "--only"])?;
            let with_deps: bool = matches.contains("--with-deps");
            let path = {
                let mut trailing = matches.free()?;
                if trailing.len() != 1 {
                    eprintln!("{}", help::ANALYSIS_STATS_HELP);
                    Err("Invalid flags")?;
                }
                trailing.pop().unwrap()
            };
            analysis_stats::run(
                verbosity,
                memory_usage,
                path.as_ref(),
                only.as_ref().map(String::as_ref),
                with_deps,
                randomize,
            )?;
        }
        "analysis-bench" => {
            if matches.contains(["-h", "--help"]) {
                eprintln!("{}", help::ANALYSIS_BENCH_HELP);
                return Ok(());
            }
            let verbose = matches.contains(["-v", "--verbose"]);
            let path: String = matches.opt_value_from_str("--path")?.unwrap_or_default();
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
            matches.finish().or_else(handle_extra_flags)?;
            analysis_bench::run(verbose, path.as_ref(), op)?;
        }
        _ => eprintln!("{}", help::GLOBAL_HELP),
    }
    Ok(())
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
