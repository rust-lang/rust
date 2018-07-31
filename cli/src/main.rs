extern crate clap;
#[macro_use]
extern crate failure;
extern crate libsyntax2;
extern crate tools;

use clap::{App, Arg, SubCommand};
use std::time::Instant;
use std::{fs, io::Read, path::Path};
use tools::collect_tests;

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
        .subcommand(SubCommand::with_name("parse"))
        .get_matches();
    match matches.subcommand() {
        ("parse", _) => {
            let tree = parse()?;
            println!("{}", tree);
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

fn parse() -> Result<String> {
    let text = read_stdin()?;
    let start = Instant::now();
    let file = libsyntax2::parse(text);
    eprintln!("elapsed {:?}", start.elapsed());
    let tree = libsyntax2::utils::dump_tree(&file);
    Ok(tree)
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
    let file = libsyntax2::parse(test.text.clone());
    let tree = libsyntax2::utils::dump_tree(&file);
    Ok((test.text, tree))
}
