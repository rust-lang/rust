#![feature(rustc_private)]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]

use clap::{Args, Parser, Subcommand};
use clippy_dev::{dogfood, fmt, lint, new_lint, release, serve, setup, sync, update_lints, utils};
use std::convert::Infallible;

fn main() {
    let dev = Dev::parse();

    match dev.command {
        DevCommand::Bless => {
            eprintln!("use `cargo bless` to automatically replace `.stderr` and `.fixed` files as tests are being run");
        },
        DevCommand::Dogfood {
            fix,
            allow_dirty,
            allow_staged,
        } => dogfood::dogfood(fix, allow_dirty, allow_staged),
        DevCommand::Fmt { check, verbose } => fmt::run(check, verbose),
        DevCommand::UpdateLints { print_only, check } => {
            if print_only {
                update_lints::print_lints();
            } else if check {
                update_lints::update(utils::UpdateMode::Check);
            } else {
                update_lints::update(utils::UpdateMode::Change);
            }
        },
        DevCommand::NewLint {
            pass,
            name,
            category,
            r#type,
            msrv,
        } => match new_lint::create(&pass, &name, &category, r#type.as_deref(), msrv) {
            Ok(()) => update_lints::update(utils::UpdateMode::Change),
            Err(e) => eprintln!("Unable to create lint: {e}"),
        },
        DevCommand::Setup(SetupCommand { subcommand }) => match subcommand {
            SetupSubcommand::Intellij { remove, repo_path } => {
                if remove {
                    setup::intellij::remove_rustc_src();
                } else {
                    setup::intellij::setup_rustc_src(&repo_path);
                }
            },
            SetupSubcommand::GitHook { remove, force_override } => {
                if remove {
                    setup::git_hook::remove_hook();
                } else {
                    setup::git_hook::install_hook(force_override);
                }
            },
            SetupSubcommand::Toolchain { force, release, name } => setup::toolchain::create(force, release, &name),
            SetupSubcommand::VscodeTasks { remove, force_override } => {
                if remove {
                    setup::vscode::remove_tasks();
                } else {
                    setup::vscode::install_tasks(force_override);
                }
            },
        },
        DevCommand::Remove(RemoveCommand { subcommand }) => match subcommand {
            RemoveSubcommand::Intellij => setup::intellij::remove_rustc_src(),
            RemoveSubcommand::GitHook => setup::git_hook::remove_hook(),
            RemoveSubcommand::VscodeTasks => setup::vscode::remove_tasks(),
        },
        DevCommand::Serve { port, lint } => serve::run(port, lint),
        DevCommand::Lint { path, args } => lint::run(&path, args.iter()),
        DevCommand::RenameLint {
            old_name,
            new_name,
            uplift,
        } => update_lints::rename(&old_name, new_name.as_ref().unwrap_or(&old_name), uplift),
        DevCommand::Deprecate { name, reason } => update_lints::deprecate(&name, &reason),
        DevCommand::Sync(SyncCommand { subcommand }) => match subcommand {
            SyncSubcommand::UpdateNightly => sync::update_nightly(),
        },
        DevCommand::Release(ReleaseCommand { subcommand }) => match subcommand {
            ReleaseSubcommand::BumpVersion => release::bump_version(),
        },
    }
}

#[derive(Parser)]
#[command(name = "dev", about)]
struct Dev {
    #[command(subcommand)]
    command: DevCommand,
}

#[derive(Subcommand)]
enum DevCommand {
    /// Bless the test output changes
    Bless,
    /// Runs the dogfood test
    Dogfood {
        #[arg(long)]
        /// Apply the suggestions when possible
        fix: bool,
        #[arg(long, requires = "fix")]
        /// Fix code even if the working directory has changes
        allow_dirty: bool,
        #[arg(long, requires = "fix")]
        /// Fix code even if the working directory has staged changes
        allow_staged: bool,
    },
    /// Run rustfmt on all projects and tests
    Fmt {
        #[arg(long)]
        /// Use the rustfmt --check option
        check: bool,
        #[arg(short, long)]
        /// Echo commands run
        verbose: bool,
    },
    #[command(name = "update_lints")]
    /// Updates lint registration and information from the source code
    ///
    /// Makes sure that: {n}
    /// * the lint count in README.md is correct {n}
    /// * the changelog contains markdown link references at the bottom {n}
    /// * all lint groups include the correct lints {n}
    /// * lint modules in `clippy_lints/*` are visible in `src/lib.rs` via `pub mod` {n}
    /// * all lints are registered in the lint store
    UpdateLints {
        #[arg(long)]
        /// Print a table of lints to STDOUT
        ///
        /// This does not include deprecated and internal lints. (Does not modify any files)
        print_only: bool,
        #[arg(long)]
        /// Checks that `cargo dev update_lints` has been run. Used on CI.
        check: bool,
    },
    #[command(name = "new_lint")]
    /// Create a new lint and run `cargo dev update_lints`
    NewLint {
        #[arg(short, long, value_parser = ["early", "late"], conflicts_with = "type", default_value = "late")]
        /// Specify whether the lint runs during the early or late pass
        pass: String,
        #[arg(
            short,
            long,
            value_parser = |name: &str| Ok::<_, Infallible>(name.replace('-', "_")),
        )]
        /// Name of the new lint in snake case, ex: `fn_too_long`
        name: String,
        #[arg(
            short,
            long,
            value_parser = [
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
            ],
            default_value = "nursery",
        )]
        /// What category the lint belongs to
        category: String,
        #[arg(long)]
        /// What directory the lint belongs in
        r#type: Option<String>,
        #[arg(long)]
        /// Add MSRV config code to the lint
        msrv: bool,
    },
    /// Support for setting up your personal development environment
    Setup(SetupCommand),
    /// Support for removing changes done by the setup command
    Remove(RemoveCommand),
    /// Launch a local 'ALL the Clippy Lints' website in a browser
    Serve {
        #[arg(short, long, default_value = "8000")]
        /// Local port for the http server
        port: u16,
        #[arg(long)]
        /// Which lint's page to load initially (optional)
        lint: Option<String>,
    },
    #[allow(clippy::doc_markdown)]
    /// Manually run clippy on a file or package
    ///
    /// ## Examples
    ///
    /// Lint a single file: {n}
    ///     cargo dev lint tests/ui/attrs.rs
    ///
    /// Lint a package directory: {n}
    ///     cargo dev lint tests/ui-cargo/wildcard_dependencies/fail {n}
    ///     cargo dev lint ~/my-project
    ///
    /// Run rustfix: {n}
    ///     cargo dev lint ~/my-project -- --fix
    ///
    /// Set lint levels: {n}
    ///     cargo dev lint file.rs -- -W clippy::pedantic {n}
    ///     cargo dev lint ~/my-project -- -- -W clippy::pedantic
    Lint {
        /// The path to a file or package directory to lint
        path: String,
        /// Pass extra arguments to cargo/clippy-driver
        args: Vec<String>,
    },
    #[command(name = "rename_lint")]
    /// Rename a lint
    RenameLint {
        /// The name of the lint to rename
        old_name: String,
        #[arg(required_unless_present = "uplift")]
        /// The new name of the lint
        new_name: Option<String>,
        #[arg(long)]
        /// This lint will be uplifted into rustc
        uplift: bool,
    },
    /// Deprecate the given lint
    Deprecate {
        /// The name of the lint to deprecate
        name: String,
        #[arg(long, short)]
        /// The reason for deprecation
        reason: String,
    },
    /// Sync between the rust repo and the Clippy repo
    Sync(SyncCommand),
    /// Manage Clippy releases
    Release(ReleaseCommand),
}

#[derive(Args)]
struct SetupCommand {
    #[command(subcommand)]
    subcommand: SetupSubcommand,
}

#[derive(Subcommand)]
enum SetupSubcommand {
    /// Alter dependencies so Intellij Rust can find rustc internals
    Intellij {
        #[arg(long)]
        /// Remove the dependencies added with 'cargo dev setup intellij'
        remove: bool,
        #[arg(long, short, conflicts_with = "remove")]
        /// The path to a rustc repo that will be used for setting the dependencies
        repo_path: String,
    },
    /// Add a pre-commit git hook that formats your code to make it look pretty
    GitHook {
        #[arg(long)]
        /// Remove the pre-commit hook added with 'cargo dev setup git-hook'
        remove: bool,
        #[arg(long, short)]
        /// Forces the override of an existing git pre-commit hook
        force_override: bool,
    },
    /// Install a rustup toolchain pointing to the local clippy build
    Toolchain {
        #[arg(long, short)]
        /// Override an existing toolchain
        force: bool,
        #[arg(long, short)]
        /// Point to --release clippy binary
        release: bool,
        #[arg(long, default_value = "clippy")]
        /// Name of the toolchain
        name: String,
    },
    /// Add several tasks to vscode for formatting, validation and testing
    VscodeTasks {
        #[arg(long)]
        /// Remove the tasks added with 'cargo dev setup vscode-tasks'
        remove: bool,
        #[arg(long, short)]
        /// Forces the override of existing vscode tasks
        force_override: bool,
    },
}

#[derive(Args)]
struct RemoveCommand {
    #[command(subcommand)]
    subcommand: RemoveSubcommand,
}

#[derive(Subcommand)]
enum RemoveSubcommand {
    /// Remove the dependencies added with 'cargo dev setup intellij'
    Intellij,
    /// Remove the pre-commit git hook
    GitHook,
    /// Remove the tasks added with 'cargo dev setup vscode-tasks'
    VscodeTasks,
}

#[derive(Args)]
struct SyncCommand {
    #[command(subcommand)]
    subcommand: SyncSubcommand,
}

#[derive(Subcommand)]
enum SyncSubcommand {
    #[command(name = "update_nightly")]
    /// Update nightly version in rust-toolchain and `clippy_utils`
    UpdateNightly,
}

#[derive(Args)]
struct ReleaseCommand {
    #[command(subcommand)]
    subcommand: ReleaseSubcommand,
}

#[derive(Subcommand)]
enum ReleaseSubcommand {
    #[command(name = "bump_version")]
    /// Bump the version in the Cargo.toml files
    BumpVersion,
}
