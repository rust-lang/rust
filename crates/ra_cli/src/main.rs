mod analysis_stats;
mod analysis_bench;

use std::{io::Read, error::Error};

use clap::{App, Arg, SubCommand};
use ra_ide_api::{file_structure, Analysis};
use ra_syntax::{SourceFile, TreeArc, AstNode};
use flexi_logger::Logger;
use ra_prof::profile;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> Result<()> {
    Logger::with_env().start()?;
    let matches = App::new("ra-cli")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .subcommand(SubCommand::with_name("parse").arg(Arg::with_name("no-dump").long("--no-dump")))
        .subcommand(SubCommand::with_name("symbols"))
        .subcommand(
            SubCommand::with_name("highlight")
                .arg(Arg::with_name("rainbow").short("r").long("rainbow")),
        )
        .subcommand(
            SubCommand::with_name("analysis-stats")
                .arg(Arg::with_name("verbose").short("v").long("verbose"))
                .arg(Arg::with_name("only").short("o").takes_value(true))
                .arg(Arg::with_name("path")),
        )
        .subcommand(
            SubCommand::with_name("analysis-bench")
                .arg(Arg::with_name("verbose").short("v").long("verbose"))
                .arg(
                    Arg::with_name("highlight")
                        .long("highlight")
                        .takes_value(true)
                        .conflicts_with("complete")
                        .value_name("PATH")
                        .help("highlight this file"),
                )
                .arg(
                    Arg::with_name("complete")
                        .long("complete")
                        .takes_value(true)
                        .conflicts_with("highlight")
                        .value_name("PATH:LINE:COLUMN")
                        .help("compute completions at this location"),
                )
                .arg(Arg::with_name("path").value_name("PATH").help("project to analyze")),
        )
        .get_matches();
    match matches.subcommand() {
        ("parse", Some(matches)) => {
            let _p = profile("parsing");
            let file = file()?;
            if !matches.is_present("no-dump") {
                println!("{}", file.syntax().debug_dump());
            }
            std::mem::forget(file);
        }
        ("symbols", _) => {
            let file = file()?;
            for s in file_structure(&file) {
                println!("{:?}", s);
            }
        }
        ("highlight", Some(matches)) => {
            let (analysis, file_id) = Analysis::from_single_file(read_stdin()?);
            let html = analysis.highlight_as_html(file_id, matches.is_present("rainbow")).unwrap();
            println!("{}", html);
        }
        ("analysis-stats", Some(matches)) => {
            let verbose = matches.is_present("verbose");
            let path = matches.value_of("path").unwrap_or("");
            let only = matches.value_of("only");
            analysis_stats::run(verbose, path.as_ref(), only)?;
        }
        ("analysis-bench", Some(matches)) => {
            let verbose = matches.is_present("verbose");
            let path = matches.value_of("path").unwrap_or("");
            let op = if let Some(path) = matches.value_of("highlight") {
                analysis_bench::Op::Highlight { path: path.into() }
            } else if let Some(path_line_col) = matches.value_of("complete") {
                let (path_line, column) = rsplit_at_char(path_line_col, ':')?;
                let (path, line) = rsplit_at_char(path_line, ':')?;
                analysis_bench::Op::Complete {
                    path: path.into(),
                    line: line.parse()?,
                    column: column.parse()?,
                }
            } else {
                panic!("either --highlight or --complete must be set")
            };
            analysis_bench::run(verbose, path.as_ref(), op)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn file() -> Result<TreeArc<SourceFile>> {
    let text = read_stdin()?;
    Ok(SourceFile::parse(&text).tree)
}

fn read_stdin() -> Result<String> {
    let mut buff = String::new();
    std::io::stdin().read_to_string(&mut buff)?;
    Ok(buff)
}

fn rsplit_at_char(s: &str, c: char) -> Result<(&str, &str)> {
    let idx = s.rfind(":").ok_or_else(|| format!("no `{}` in {}", c, s))?;
    Ok((&s[..idx], &s[idx + 1..]))
}
