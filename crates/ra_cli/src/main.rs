mod analysis_stats;

use std::{fs, io::Read, path::Path};

use clap::{App, Arg, SubCommand};
use ra_ide_api::file_structure;
use ra_syntax::{SourceFile, TreeArc, AstNode};
use tools::collect_tests;
use flexi_logger::Logger;
use ra_prof::profile;

type Result<T> = ::std::result::Result<T, failure::Error>;

fn main() -> Result<()> {
    Logger::with_env().start()?;
    let matches = App::new("ra-cli")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .subcommand(
            SubCommand::with_name("render-test")
                .arg(Arg::with_name("line").long("--line").required(true).takes_value(true))
                .arg(Arg::with_name("file").long("--file").required(true).takes_value(true)),
        )
        .subcommand(SubCommand::with_name("parse").arg(Arg::with_name("no-dump").long("--no-dump")))
        .subcommand(SubCommand::with_name("symbols"))
        .subcommand(
            SubCommand::with_name("analysis-stats")
                .arg(Arg::with_name("verbose").short("v"))
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
        ("render-test", Some(matches)) => {
            let file = matches.value_of("file").unwrap();
            let file = Path::new(file);
            let line: usize = matches.value_of("line").unwrap().parse()?;
            let line = line - 1;
            let (test, tree) = render_test(file, line)?;
            println!("{}\n{}", test, tree);
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

fn render_test(file: &Path, line: usize) -> Result<(String, String)> {
    let text = fs::read_to_string(file)?;
    let tests = collect_tests(&text);
    let test = tests.into_iter().find(|(start_line, t)| {
        *start_line <= line && line <= *start_line + t.text.lines().count()
    });
    let test = match test {
        None => failure::bail!("No test found at line {} at {}", line, file.display()),
        Some((_start_line, test)) => test,
    };
    let file = SourceFile::parse(&test.text);
    let tree = file.syntax().debug_dump();
    Ok((test.text, tree))
}
