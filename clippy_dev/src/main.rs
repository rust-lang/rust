#![cfg_attr(feature = "deny-warnings", deny(warnings))]

use clap::{App, Arg, SubCommand};
use clippy_dev::{
    gather_all, gen_changelog_lint_list, gen_deprecated, gen_lint_group_list, gen_modules_list, gen_register_lint_list,
    replace_region_in_file, Lint, DOCS_LINK,
};
use std::path::Path;

mod fmt;
mod new_lint;
mod stderr_length_check;

#[derive(Clone, Copy, PartialEq)]
enum UpdateMode {
    Check,
    Change,
}

fn main() {
    let matches = App::new("Clippy developer tooling")
        .subcommand(
            SubCommand::with_name("fmt")
                .about("Run rustfmt on all projects and tests")
                .arg(
                    Arg::with_name("check")
                        .long("check")
                        .help("Use the rustfmt --check option"),
                )
                .arg(
                    Arg::with_name("verbose")
                        .short("v")
                        .long("verbose")
                        .help("Echo commands run"),
                ),
        )
        .subcommand(
            SubCommand::with_name("update_lints")
                .about("Updates lint registration and information from the source code")
                .long_about(
                    "Makes sure that:\n \
                     * the lint count in README.md is correct\n \
                     * the changelog contains markdown link references at the bottom\n \
                     * all lint groups include the correct lints\n \
                     * lint modules in `clippy_lints/*` are visible in `src/lib.rs` via `pub mod`\n \
                     * all lints are registered in the lint store",
                )
                .arg(Arg::with_name("print-only").long("print-only").help(
                    "Print a table of lints to STDOUT. \
                     This does not include deprecated and internal lints. \
                     (Does not modify any files)",
                ))
                .arg(
                    Arg::with_name("check")
                        .long("check")
                        .help("Checks that `cargo dev update_lints` has been run. Used on CI."),
                ),
        )
        .subcommand(
            SubCommand::with_name("new_lint")
                .about("Create new lint and run `cargo dev update_lints`")
                .arg(
                    Arg::with_name("pass")
                        .short("p")
                        .long("pass")
                        .help("Specify whether the lint runs during the early or late pass")
                        .takes_value(true)
                        .possible_values(&["early", "late"])
                        .required(true),
                )
                .arg(
                    Arg::with_name("name")
                        .short("n")
                        .long("name")
                        .help("Name of the new lint in snake case, ex: fn_too_long")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::with_name("category")
                        .short("c")
                        .long("category")
                        .help("What category the lint belongs to")
                        .default_value("nursery")
                        .possible_values(&[
                            "style",
                            "correctness",
                            "complexity",
                            "perf",
                            "pedantic",
                            "restriction",
                            "cargo",
                            "nursery",
                            "internal",
                            "internal_warn",
                        ])
                        .takes_value(true),
                ),
        )
        .arg(
            Arg::with_name("limit-stderr-length")
                .long("limit-stderr-length")
                .help("Ensures that stderr files do not grow longer than a certain amount of lines."),
        )
        .get_matches();

    if matches.is_present("limit-stderr-length") {
        stderr_length_check::check();
    }

    match matches.subcommand() {
        ("fmt", Some(matches)) => {
            fmt::run(matches.is_present("check"), matches.is_present("verbose"));
        },
        ("update_lints", Some(matches)) => {
            if matches.is_present("print-only") {
                print_lints();
            } else if matches.is_present("check") {
                update_lints(UpdateMode::Check);
            } else {
                update_lints(UpdateMode::Change);
            }
        },
        ("new_lint", Some(matches)) => {
            match new_lint::create(
                matches.value_of("pass"),
                matches.value_of("name"),
                matches.value_of("category"),
            ) {
                Ok(_) => update_lints(UpdateMode::Change),
                Err(e) => eprintln!("Unable to create lint: {}", e),
            }
        },
        _ => {},
    }
}

fn print_lints() {
    let lint_list = gather_all();
    let usable_lints: Vec<Lint> = Lint::usable_lints(lint_list).collect();
    let usable_lint_count = usable_lints.len();
    let grouped_by_lint_group = Lint::by_lint_group(usable_lints.into_iter());

    for (lint_group, mut lints) in grouped_by_lint_group {
        if lint_group == "Deprecated" {
            continue;
        }
        println!("\n## {}", lint_group);

        lints.sort_by_key(|l| l.name.clone());

        for lint in lints {
            println!(
                "* [{}]({}#{}) ({})",
                lint.name,
                clippy_dev::DOCS_LINK,
                lint.name,
                lint.desc
            );
        }
    }

    println!("there are {} lints", usable_lint_count);
}

#[allow(clippy::too_many_lines)]
fn update_lints(update_mode: UpdateMode) {
    let lint_list: Vec<Lint> = gather_all().collect();

    let internal_lints = Lint::internal_lints(lint_list.clone().into_iter());

    let usable_lints: Vec<Lint> = Lint::usable_lints(lint_list.clone().into_iter()).collect();
    let usable_lint_count = usable_lints.len();

    let mut sorted_usable_lints = usable_lints.clone();
    sorted_usable_lints.sort_by_key(|lint| lint.name.clone());

    let mut file_change = replace_region_in_file(
        Path::new("src/lintlist/mod.rs"),
        "begin lint list",
        "end lint list",
        false,
        update_mode == UpdateMode::Change,
        || {
            format!(
                "pub const ALL_LINTS: [Lint; {}] = {:#?};",
                sorted_usable_lints.len(),
                sorted_usable_lints
            )
            .lines()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
        },
    )
    .changed;

    file_change |= replace_region_in_file(
        Path::new("README.md"),
        &format!(r#"\[There are \d+ lints included in this crate!\]\({}\)"#, DOCS_LINK),
        "",
        true,
        update_mode == UpdateMode::Change,
        || {
            vec![format!(
                "[There are {} lints included in this crate!]({})",
                usable_lint_count, DOCS_LINK
            )]
        },
    )
    .changed;

    file_change |= replace_region_in_file(
        Path::new("CHANGELOG.md"),
        "<!-- begin autogenerated links to lint list -->",
        "<!-- end autogenerated links to lint list -->",
        false,
        update_mode == UpdateMode::Change,
        || gen_changelog_lint_list(lint_list.clone()),
    )
    .changed;

    file_change |= replace_region_in_file(
        Path::new("clippy_lints/src/lib.rs"),
        "begin deprecated lints",
        "end deprecated lints",
        false,
        update_mode == UpdateMode::Change,
        || gen_deprecated(&lint_list),
    )
    .changed;

    file_change |= replace_region_in_file(
        Path::new("clippy_lints/src/lib.rs"),
        "begin register lints",
        "end register lints",
        false,
        update_mode == UpdateMode::Change,
        || gen_register_lint_list(&lint_list),
    )
    .changed;

    file_change |= replace_region_in_file(
        Path::new("clippy_lints/src/lib.rs"),
        "begin lints modules",
        "end lints modules",
        false,
        update_mode == UpdateMode::Change,
        || gen_modules_list(lint_list.clone()),
    )
    .changed;

    // Generate lists of lints in the clippy::all lint group
    file_change |= replace_region_in_file(
        Path::new("clippy_lints/src/lib.rs"),
        r#"store.register_group\(true, "clippy::all""#,
        r#"\]\);"#,
        false,
        update_mode == UpdateMode::Change,
        || {
            // clippy::all should only include the following lint groups:
            let all_group_lints = usable_lints
                .clone()
                .into_iter()
                .filter(|l| {
                    l.group == "correctness" || l.group == "style" || l.group == "complexity" || l.group == "perf"
                })
                .collect();

            gen_lint_group_list(all_group_lints)
        },
    )
    .changed;

    // Generate the list of lints for all other lint groups
    for (lint_group, lints) in Lint::by_lint_group(usable_lints.into_iter().chain(internal_lints)) {
        file_change |= replace_region_in_file(
            Path::new("clippy_lints/src/lib.rs"),
            &format!("store.register_group\\(true, \"clippy::{}\"", lint_group),
            r#"\]\);"#,
            false,
            update_mode == UpdateMode::Change,
            || gen_lint_group_list(lints.clone()),
        )
        .changed;
    }

    if update_mode == UpdateMode::Check && file_change {
        println!(
            "Not all lints defined properly. \
             Please run `cargo dev update_lints` to make sure all lints are defined properly."
        );
        std::process::exit(1);
    }
}
