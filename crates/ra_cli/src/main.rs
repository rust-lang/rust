mod analysis_stats;

use std::{fs, io::Read, path::Path, time::Instant};

use clap::{App, Arg, SubCommand};
use join_to_string::join;
use ra_ide_api_light::{extend_selection, file_structure};
use ra_syntax::{SourceFile, TextRange, TreeArc, AstNode};
use tools::collect_tests;
use flexi_logger::Logger;

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
            SubCommand::with_name("extend-selection")
                .arg(Arg::with_name("start"))
                .arg(Arg::with_name("end")),
        )
        .subcommand(
            SubCommand::with_name("analysis-stats").arg(Arg::with_name("verbose").short("v")),
        )
        .get_matches();
    match matches.subcommand() {
        ("parse", Some(matches)) => {
            let start = Instant::now();
            let file = file()?;
            let elapsed = start.elapsed();
            if !matches.is_present("no-dump") {
                println!("{}", file.syntax().debug_dump());
            }
            eprintln!("parsing: {:?}", elapsed);
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
        ("extend-selection", Some(matches)) => {
            let start: u32 = matches.value_of("start").unwrap().parse()?;
            let end: u32 = matches.value_of("end").unwrap().parse()?;
            let file = file()?;
            let sels = selections(&file, start, end);
            println!("{}", sels)
        }
        ("analysis-stats", Some(matches)) => {
            let verbose = matches.is_present("verbose");
            analysis_stats::run(verbose)?;
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

fn selections(file: &SourceFile, start: u32, end: u32) -> String {
    let mut ranges = Vec::new();
    let mut cur = Some(TextRange::from_to((start - 1).into(), (end - 1).into()));
    while let Some(r) = cur {
        ranges.push(r);
        cur = extend_selection(file.syntax(), r);
    }
    let ranges = ranges
        .iter()
        .map(|r| (1 + u32::from(r.start()), 1 + u32::from(r.end())))
        .map(|(s, e)| format!("({} {})", s, e));
    join(ranges).separator(" ").surround_with("(", ")").to_string()
}
