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
    let print_clippy_lint_groups: [&str; 7] = [
        "correctness",
        "style",
        "complexity",
        "perf",
        "pedantic",
        "nursery",
        "restriction"
    ];
    // We could use itertools' group_by to make this much more concise:
    for group in &print_clippy_lint_groups {
        println!("\n## {}", group);

        let mut group_lints = Lint::in_lint_group(group, &lint_list);
        group_lints.sort_by(|a, b| a.name.cmp(&b.name));

        for lint in group_lints {
            if lint.deprecation.is_some() { continue; }
            println!("* [{}]({}#{}) ({})", lint.name, clippy_dev::DOCS_LINK.clone(), lint.name, lint.desc);
        }
    }
    println!("there are {} lints", Lint::active_lints(&lint_list).len());
}
