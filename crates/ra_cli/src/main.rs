mod analysis_stats;

use std::io::Read;

use clap::{App, Arg, SubCommand};
use ra_ide_api::{file_structure, Analysis};
use ra_syntax::{SourceFile, TreeArc, AstNode};
use flexi_logger::Logger;
use ra_prof::profile;

type Result<T> = ::std::result::Result<T, failure::Error>;

fn main() -> Result<()> {
    Logger::with_env().start()?;
    let matches = App::new("ra-cli")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .subcommand(SubCommand::with_name("parse").arg(Arg::with_name("no-dump").long("--no-dump")))
        .subcommand(SubCommand::with_name("symbols"))
        .subcommand(
            SubCommand::with_name("highlight")
                .arg(Arg::with_name("rainbow").short("r").long("rainbow"))
        )
        .subcommand(
            SubCommand::with_name("analysis-stats")
                .arg(Arg::with_name("verbose").short("v").long("verbose"))
                .arg(Arg::with_name("only").short("o").takes_value(true))
                .arg(Arg::with_name("path")),
        )
        .get_matches();
    match matches.subcommand() {
        ("parse", Some(matches)) => {
            let _p = profile("parsing");
            let file = file()?;
            if !matches.is_present("no-dump") {
                println!("{}", file.syntax().debug_dump());
            }
            ::std::mem::forget(file);
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
            analysis_stats::run(verbose, path, only)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn file() -> Result<TreeArc<SourceFile>> {
    let text = read_stdin()?;
    Ok(SourceFile::parse(&text))
}

fn read_stdin() -> Result<String> {
    let mut buff = String::new();
    ::std::io::stdin().read_to_string(&mut buff)?;
    Ok(buff)
}
