extern crate clap;
extern crate clippy_dev;
extern crate regex;

use clap::{App, Arg, SubCommand};
use clippy_dev::*;

fn main() {
    let matches = App::new("Clippy developer tooling")
        .subcommand(
            SubCommand::with_name("update_lints")
                .about("Update the lint list")
                .arg(
                    Arg::with_name("print-only")
                        .long("print-only")
                        .short("p")
                        .help("Print a table of lints to STDOUT. Does not modify any files."),
                )
        )
        .get_matches();

    if let Some(matches) = matches.subcommand_matches("update_lints") {
        if matches.is_present("print-only") {
            print_lints();
        }
    }
}

fn print_lints() {
    let lint_list = collect_all();
    let grouped_by_lint_group = Lint::by_lint_group(&lint_list);

    for (lint_group, mut lints) in grouped_by_lint_group {
        if lint_group == "Deprecated" { continue; }
        println!("\n## {}", lint_group);

        lints.sort_by(|a, b| a.name.cmp(&b.name));

        for lint in lints {
            println!("* [{}]({}#{}) ({})", lint.name, clippy_dev::DOCS_LINK.clone(), lint.name, lint.desc);
        }
    }

    println!("there are {} lints", Lint::active_lints(&lint_list).len());
}
