#![cfg_attr(feature = "deny-warnings", deny(warnings))]

use clap::{App, Arg, SubCommand};
use clippy_dev::{fmt, new_lint, ra_setup, serve, stderr_length_check, update_lints};

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
        .subcommand(
            SubCommand::with_name("limit_stderr_length")
                .about("Ensures that stderr files do not grow longer than a certain amount of lines."),
        )
        .subcommand(
            SubCommand::with_name("ra-setup")
                .about("Alter dependencies so rust-analyzer can find rustc internals")
                .arg(
                    Arg::with_name("rustc-repo-path")
                        .long("repo-path")
                        .short("r")
                        .help("The path to a rustc repo that will be used for setting the dependencies")
                        .takes_value(true)
                        .value_name("path")
                        .required(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("serve")
                .about("Launch a local 'ALL the Clippy Lints' website in a browser")
                .arg(
                    Arg::with_name("port")
                        .long("port")
                        .short("p")
                        .help("Local port for the http server")
                        .default_value("8000")
                        .validator_os(serve::validate_port),
                )
                .arg(Arg::with_name("lint").help("Which lint's page to load initially (optional)")),
        )
        .get_matches();

    match matches.subcommand() {
        ("fmt", Some(matches)) => {
            fmt::run(matches.is_present("check"), matches.is_present("verbose"));
        },
        ("update_lints", Some(matches)) => {
            if matches.is_present("print-only") {
                update_lints::print_lints();
            } else if matches.is_present("check") {
                update_lints::run(update_lints::UpdateMode::Check);
            } else {
                update_lints::run(update_lints::UpdateMode::Change);
            }
        },
        ("new_lint", Some(matches)) => {
            match new_lint::create(
                matches.value_of("pass"),
                matches.value_of("name"),
                matches.value_of("category"),
            ) {
                Ok(_) => update_lints::run(update_lints::UpdateMode::Change),
                Err(e) => eprintln!("Unable to create lint: {}", e),
            }
        },
        ("limit_stderr_length", _) => {
            stderr_length_check::check();
        },
        ("ra-setup", Some(matches)) => ra_setup::run(matches.value_of("rustc-repo-path")),
        ("serve", Some(matches)) => {
            let port = matches.value_of("port").unwrap().parse().unwrap();
            let lint = matches.value_of("lint");
            serve::run(port, lint);
        },
        _ => {},
    }
}
