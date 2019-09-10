mod analysis_stats;
mod analysis_bench;
mod help;

use std::{error::Error, fmt::Write, io::Read};

use flexi_logger::Logger;
use pico_args::Arguments;
use ra_ide_api::{file_structure, Analysis};
use ra_prof::profile;
use ra_syntax::{AstNode, SourceFile};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> Result<()> {
    Logger::with_env().start()?;

    let subcommand = std::env::args_os().nth(1);
    if subcommand.is_none() {
        help::print_global_help();
        return Ok(());
    }
    let subcommand = subcommand.unwrap();
    let mut matches = Arguments::from_vec(std::env::args_os().skip(2).collect());

    match &*subcommand.to_string_lossy() {
        "parse" => {
            if matches.contains(["-h", "--help"]) {
                help::print_parse_help();
                return Ok(());
            } else {
                let no_dump = matches.contains("--no-dump");
                matches.finish().or_else(handle_extra_flags)?;

                let _p = profile("parsing");
                let file = file()?;
                if !no_dump {
                    println!("{:#?}", file.syntax());
                }
                std::mem::forget(file);
            }
        }
        "symbols" => {
            if matches.contains(["-h", "--help"]) {
                help::print_symbols_help();
                return Ok(());
            } else {
                matches.finish().or_else(handle_extra_flags)?;
                let file = file()?;
                for s in file_structure(&file) {
                    println!("{:?}", s);
                }
            }
        }
        "highlight" => {
            if matches.contains(["-h", "--help"]) {
                help::print_highlight_help();
                return Ok(());
            } else {
                let rainbow_opt = matches.contains(["-r", "--rainbow"]);
                matches.finish().or_else(handle_extra_flags)?;
                let (analysis, file_id) = Analysis::from_single_file(read_stdin()?);
                let html = analysis.highlight_as_html(file_id, rainbow_opt).unwrap();
                println!("{}", html);
            }
        }
        "analysis-stats" => {
            if matches.contains(["-h", "--help"]) {
                help::print_analysis_stats_help();
                return Ok(());
            } else {
                let verbose = matches.contains(["-v", "--verbose"]);
                let memory_usage = matches.contains("--memory-usage");
                let path = matches.value_from_str("--path")?.unwrap_or("".to_string());
                let only = matches.value_from_str(["-o", "--only"])?.map(|v: String| v.to_owned());
                matches.finish().or_else(handle_extra_flags)?;
                analysis_stats::run(
                    verbose,
                    memory_usage,
                    path.as_ref(),
                    only.as_ref().map(String::as_ref),
                )?;
            }
        }
        "analysis-bench" => {
            if matches.contains(["-h", "--help"]) {
                help::print_analysis_bench_help();
                return Ok(());
            } else {
                let verbose = matches.contains(["-v", "--verbose"]);
                let path = matches.value_from_str("--path")?.unwrap_or("".to_string());
                let highlight_path = matches.value_from_str("--highlight")?;
                let complete_path = matches.value_from_str("--complete")?;
                if highlight_path.is_some() && complete_path.is_some() {
                    panic!("either --highlight or --complete must be set, not both")
                }
                let op = if let Some(path) = highlight_path {
                    let path: String = path;
                    analysis_bench::Op::Highlight { path: path.into() }
                } else if let Some(path_line_col) = complete_path {
                    let path_line_col: String = path_line_col;
                    let (path_line, column) = rsplit_at_char(path_line_col.as_str(), ':')?;
                    let (path, line) = rsplit_at_char(path_line, ':')?;
                    analysis_bench::Op::Complete {
                        path: path.into(),
                        line: line.parse()?,
                        column: column.parse()?,
                    }
                } else {
                    panic!("either --highlight or --complete must be set")
                };
                matches.finish().or_else(handle_extra_flags)?;
                analysis_bench::run(verbose, path.as_ref(), op)?;
            }
        }
        _ => help::print_global_help(),
    }
    Ok(())
}

fn handle_extra_flags(e: pico_args::Error) -> Result<()> {
    if let pico_args::Error::UnusedArgsLeft(flags) = e {
        let mut invalid_flags = String::new();
        for flag in flags {
            write!(&mut invalid_flags, "{}, ", flag).expect("Error on write");
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

fn rsplit_at_char(s: &str, c: char) -> Result<(&str, &str)> {
    let idx = s.rfind(':').ok_or_else(|| format!("no `{}` in {}", c, s))?;
    Ok((&s[..idx], &s[idx + 1..]))
}
