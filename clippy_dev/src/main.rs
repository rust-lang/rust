#![cfg_attr(feature = "deny-warnings", deny(warnings))]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]

use clap::{Arg, ArgAction, ArgMatches, Command};
use clippy_dev::{bless, dogfood, fmt, lint, new_lint, serve, setup, update_lints};
use indoc::indoc;

fn main() {
    let matches = get_clap_config();

    match matches.subcommand() {
        Some(("bless", matches)) => {
            bless::bless(matches.get_flag("ignore-timestamp"));
        },
        Some(("dogfood", matches)) => {
            dogfood::dogfood(
                matches.get_flag("fix"),
                matches.get_flag("allow-dirty"),
                matches.get_flag("allow-staged"),
            );
        },
        Some(("fmt", matches)) => {
            fmt::run(matches.get_flag("check"), matches.get_flag("verbose"));
        },
        Some(("update_lints", matches)) => {
            if matches.get_flag("print-only") {
                update_lints::print_lints();
            } else if matches.get_flag("check") {
                update_lints::update(update_lints::UpdateMode::Check);
            } else {
                update_lints::update(update_lints::UpdateMode::Change);
            }
        },
        Some(("new_lint", matches)) => {
            match new_lint::create(
                matches.get_one::<String>("pass"),
                matches.get_one::<String>("name"),
                matches.get_one::<String>("category").map(String::as_str),
                matches.get_one::<String>("type").map(String::as_str),
                matches.get_flag("msrv"),
            ) {
                Ok(_) => update_lints::update(update_lints::UpdateMode::Change),
                Err(e) => eprintln!("Unable to create lint: {e}"),
            }
        },
        Some(("setup", sub_command)) => match sub_command.subcommand() {
            Some(("intellij", matches)) => {
                if matches.get_flag("remove") {
                    setup::intellij::remove_rustc_src();
                } else {
                    setup::intellij::setup_rustc_src(
                        matches
                            .get_one::<String>("rustc-repo-path")
                            .expect("this field is mandatory and therefore always valid"),
                    );
                }
            },
            Some(("git-hook", matches)) => {
                if matches.get_flag("remove") {
                    setup::git_hook::remove_hook();
                } else {
                    setup::git_hook::install_hook(matches.get_flag("force-override"));
                }
            },
            Some(("vscode-tasks", matches)) => {
                if matches.get_flag("remove") {
                    setup::vscode::remove_tasks();
                } else {
                    setup::vscode::install_tasks(matches.get_flag("force-override"));
                }
            },
            _ => {},
        },
        Some(("remove", sub_command)) => match sub_command.subcommand() {
            Some(("git-hook", _)) => setup::git_hook::remove_hook(),
            Some(("intellij", _)) => setup::intellij::remove_rustc_src(),
            Some(("vscode-tasks", _)) => setup::vscode::remove_tasks(),
            _ => {},
        },
        Some(("serve", matches)) => {
            let port = *matches.get_one::<u16>("port").unwrap();
            let lint = matches.get_one::<String>("lint");
            serve::run(port, lint);
        },
        Some(("lint", matches)) => {
            let path = matches.get_one::<String>("path").unwrap();
            let args = matches.get_many::<String>("args").into_iter().flatten();
            lint::run(path, args);
        },
        Some(("rename_lint", matches)) => {
            let old_name = matches.get_one::<String>("old_name").unwrap();
            let new_name = matches.get_one::<String>("new_name").unwrap_or(old_name);
            let uplift = matches.get_flag("uplift");
            update_lints::rename(old_name, new_name, uplift);
        },
        Some(("deprecate", matches)) => {
            let name = matches.get_one::<String>("name").unwrap();
            let reason = matches.get_one("reason");
            update_lints::deprecate(name, reason);
        },
        _ => {},
    }
}

fn get_clap_config() -> ArgMatches {
    Command::new("Clippy developer tooling")
        .arg_required_else_help(true)
        .subcommands([
            Command::new("bless").about("bless the test output changes").arg(
                Arg::new("ignore-timestamp")
                    .long("ignore-timestamp")
                    .action(ArgAction::SetTrue)
                    .help("Include files updated before clippy was built"),
            ),
            Command::new("dogfood").about("Runs the dogfood test").args([
                Arg::new("fix")
                    .long("fix")
                    .action(ArgAction::SetTrue)
                    .help("Apply the suggestions when possible"),
                Arg::new("allow-dirty")
                    .long("allow-dirty")
                    .action(ArgAction::SetTrue)
                    .help("Fix code even if the working directory has changes")
                    .requires("fix"),
                Arg::new("allow-staged")
                    .long("allow-staged")
                    .action(ArgAction::SetTrue)
                    .help("Fix code even if the working directory has staged changes")
                    .requires("fix"),
            ]),
            Command::new("fmt")
                .about("Run rustfmt on all projects and tests")
                .args([
                    Arg::new("check")
                        .long("check")
                        .action(ArgAction::SetTrue)
                        .help("Use the rustfmt --check option"),
                    Arg::new("verbose")
                        .short('v')
                        .long("verbose")
                        .action(ArgAction::SetTrue)
                        .help("Echo commands run"),
                ]),
            Command::new("update_lints")
                .about("Updates lint registration and information from the source code")
                .long_about(
                    "Makes sure that:\n \
                    * the lint count in README.md is correct\n \
                    * the changelog contains markdown link references at the bottom\n \
                    * all lint groups include the correct lints\n \
                    * lint modules in `clippy_lints/*` are visible in `src/lib.rs` via `pub mod`\n \
                    * all lints are registered in the lint store",
                )
                .args([
                    Arg::new("print-only")
                        .long("print-only")
                        .action(ArgAction::SetTrue)
                        .help(
                            "Print a table of lints to STDOUT. \
                            This does not include deprecated and internal lints. \
                            (Does not modify any files)",
                        ),
                    Arg::new("check")
                        .long("check")
                        .action(ArgAction::SetTrue)
                        .help("Checks that `cargo dev update_lints` has been run. Used on CI."),
                ]),
            Command::new("new_lint")
                .about("Create new lint and run `cargo dev update_lints`")
                .args([
                    Arg::new("pass")
                        .short('p')
                        .long("pass")
                        .help("Specify whether the lint runs during the early or late pass")
                        .value_parser(["early", "late"])
                        .conflicts_with("type")
                        .required_unless_present("type"),
                    Arg::new("name")
                        .short('n')
                        .long("name")
                        .help("Name of the new lint in snake case, ex: fn_too_long")
                        .required(true),
                    Arg::new("category")
                        .short('c')
                        .long("category")
                        .help("What category the lint belongs to")
                        .default_value("nursery")
                        .value_parser([
                            "style",
                            "correctness",
                            "suspicious",
                            "complexity",
                            "perf",
                            "pedantic",
                            "restriction",
                            "cargo",
                            "nursery",
                            "internal",
                            "internal_warn",
                        ]),
                    Arg::new("type").long("type").help("What directory the lint belongs in"),
                    Arg::new("msrv")
                        .long("msrv")
                        .action(ArgAction::SetTrue)
                        .help("Add MSRV config code to the lint"),
                ]),
            Command::new("setup")
                .about("Support for setting up your personal development environment")
                .arg_required_else_help(true)
                .subcommands([
                    Command::new("intellij")
                        .about("Alter dependencies so Intellij Rust can find rustc internals")
                        .args([
                            Arg::new("remove")
                                .long("remove")
                                .action(ArgAction::SetTrue)
                                .help("Remove the dependencies added with 'cargo dev setup intellij'"),
                            Arg::new("rustc-repo-path")
                                .long("repo-path")
                                .short('r')
                                .help("The path to a rustc repo that will be used for setting the dependencies")
                                .value_name("path")
                                .conflicts_with("remove")
                                .required(true),
                        ]),
                    Command::new("git-hook")
                        .about("Add a pre-commit git hook that formats your code to make it look pretty")
                        .args([
                            Arg::new("remove")
                                .long("remove")
                                .action(ArgAction::SetTrue)
                                .help("Remove the pre-commit hook added with 'cargo dev setup git-hook'"),
                            Arg::new("force-override")
                                .long("force-override")
                                .short('f')
                                .action(ArgAction::SetTrue)
                                .help("Forces the override of an existing git pre-commit hook"),
                        ]),
                    Command::new("vscode-tasks")
                        .about("Add several tasks to vscode for formatting, validation and testing")
                        .args([
                            Arg::new("remove")
                                .long("remove")
                                .action(ArgAction::SetTrue)
                                .help("Remove the tasks added with 'cargo dev setup vscode-tasks'"),
                            Arg::new("force-override")
                                .long("force-override")
                                .short('f')
                                .action(ArgAction::SetTrue)
                                .help("Forces the override of existing vscode tasks"),
                        ]),
                ]),
            Command::new("remove")
                .about("Support for undoing changes done by the setup command")
                .arg_required_else_help(true)
                .subcommands([
                    Command::new("git-hook").about("Remove any existing pre-commit git hook"),
                    Command::new("vscode-tasks").about("Remove any existing vscode tasks"),
                    Command::new("intellij").about("Removes rustc source paths added via `cargo dev setup intellij`"),
                ]),
            Command::new("serve")
                .about("Launch a local 'ALL the Clippy Lints' website in a browser")
                .args([
                    Arg::new("port")
                        .long("port")
                        .short('p')
                        .help("Local port for the http server")
                        .default_value("8000")
                        .value_parser(clap::value_parser!(u16)),
                    Arg::new("lint").help("Which lint's page to load initially (optional)"),
                ]),
            Command::new("lint")
                .about("Manually run clippy on a file or package")
                .after_help(indoc! {"
                    EXAMPLES
                        Lint a single file:
                            cargo dev lint tests/ui/attrs.rs

                        Lint a package directory:
                            cargo dev lint tests/ui-cargo/wildcard_dependencies/fail
                            cargo dev lint ~/my-project

                        Run rustfix:
                            cargo dev lint ~/my-project -- --fix

                        Set lint levels:
                            cargo dev lint file.rs -- -W clippy::pedantic
                            cargo dev lint ~/my-project -- -- -W clippy::pedantic
                "})
                .args([
                    Arg::new("path")
                        .required(true)
                        .help("The path to a file or package directory to lint"),
                    Arg::new("args")
                        .action(ArgAction::Append)
                        .help("Pass extra arguments to cargo/clippy-driver"),
                ]),
            Command::new("rename_lint").about("Renames the given lint").args([
                Arg::new("old_name")
                    .index(1)
                    .required(true)
                    .help("The name of the lint to rename"),
                Arg::new("new_name")
                    .index(2)
                    .required_unless_present("uplift")
                    .help("The new name of the lint"),
                Arg::new("uplift")
                    .long("uplift")
                    .action(ArgAction::SetTrue)
                    .help("This lint will be uplifted into rustc"),
            ]),
            Command::new("deprecate").about("Deprecates the given lint").args([
                Arg::new("name")
                    .index(1)
                    .required(true)
                    .help("The name of the lint to deprecate"),
                Arg::new("reason")
                    .long("reason")
                    .short('r')
                    .help("The reason for deprecation"),
            ]),
        ])
        .get_matches()
}
