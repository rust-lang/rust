extern crate clap;
#[macro_use]
extern crate failure;
extern crate libeditor;
extern crate tools;

use std::{
    fs, io::Read, path::Path,
    time::Instant
};
use clap::{App, Arg, SubCommand};
use tools::collect_tests;
use libeditor::{File, syntax_tree, file_structure};

type Result<T> = ::std::result::Result<T, failure::Error>;

fn main() -> Result<()> {
    let matches = App::new("libsyntax2-cli")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .subcommand(
            SubCommand::with_name("render-test")
                .arg(
                    Arg::with_name("line")
                        .long("--line")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("file")
                        .long("--file")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("parse")
                .arg(Arg::with_name("no-dump").long("--no-dump"))
        )
        .subcommand(SubCommand::with_name("symbols"))
        .get_matches();
    match matches.subcommand() {
        ("parse", Some(matches)) => {
            let start = Instant::now();
            let file = file()?;
            let elapsed = start.elapsed();
            if !matches.is_present("no-dump") {
                println!("{}", syntax_tree(&file));
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
        _ => unreachable!(),
    }
    Ok(())
}

fn file() -> Result<File> {
    let text = read_stdin()?;
    Ok(libeditor::parse(&text))
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
        None => bail!("No test found at line {} at {}", line, file.display()),
        Some((_start_line, test)) => test,
    };
    let file = libeditor::parse(&test.text);
    let tree = syntax_tree(&file);
    Ok((test.text, tree))
}
