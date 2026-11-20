use std::io::Write;
use std::ops::ControlFlow::{Break, Continue};
use std::path::{Path, PathBuf};
use std::{fs, io};

use crate::core::build_steps::setup::profile::PROFILE_DIR;
use crate::core::build_steps::setup::{Profile, prompt_user};
use crate::core::builder::{Builder, CommandLineStep, RunConfig, ShouldRun};
use crate::core::config::Config;
use crate::utils::change_tracker::CONFIG_CHANGE_HISTORY;
use crate::utils::helpers::{self, t};

/// Create a `bootstrap.toml` with a profile specified by the user.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BootstrapToml {
    profile: Profile,
    deny_warnings: bool,
}

impl CommandLineStep for BootstrapToml {
    type Output = ();

    fn should_run(mut run: ShouldRun<'_>) -> ShouldRun<'_> {
        for choice in Profile::all() {
            // Some of the profile names happen to coincide with actual directory names
            // ("compiler" and "library"), so avoid the usual assertion that aliases
            // don't exist on disk.
            run = run.alias_without_assert(choice.as_str());
        }
        run
    }

    fn is_default_step(_builder: &Builder<'_>) -> bool {
        true
    }

    fn make_run(run: RunConfig<'_>) {
        if run.builder.config.dry_run() {
            return;
        }

        let path = &run.builder.config.config.clone().unwrap_or(PathBuf::from("bootstrap.toml"));
        if path.exists() {
            eprintln!();
            eprintln!(
                "ERROR: you asked for a new config file, but one already exists at `{}`",
                t!(path.canonicalize()).display()
            );

            let override_ = prompt_user(
                "Do you wish to override the existing configuration \
                (which will allow the setup process to continue)?: [y/N]",
                |input| match input {
                    "y" => Break(true),
                    "n" | "" => Break(false),
                    _ => Continue(()),
                },
            )
            .unwrap_or(false);

            if override_ {
                t!(fs::remove_file(path));
            } else {
                println!("Exiting.");
                helpers::exit_process(1);
            }
        }

        // for BootstrapToml, `run.paths` will have 1 and only 1 element
        // this is because we only accept at most 1 path from user input.
        // If user calls `x.py setup` without arguments, the interactive TUI
        // will guide user to provide one.
        let profile = if run.paths.len() > 1 {
            // HACK: `builder` runs this step with all paths if no path was passed.
            t!(interactive_path())
        } else {
            run.paths
                .first()
                .unwrap()
                .assert_single_path()
                .path
                .as_path()
                .as_os_str()
                .to_str()
                .unwrap()
                .parse()
                .unwrap()
        };

        let deny_warnings = {
            println!(
                "\nRust CI fails when there are any warnings. It is possible to deny warnings locally too, \
                 but that can hurt iteration cycle (forcing you to rebuild everything when there are warnings).\n"
            );

            prompt_user("Deny warnings? [y/N]", |input| match input {
                "y" => Break(true),
                "n" | "" => Break(false),
                _ => Continue(()),
            })
            .unwrap()
        };

        run.builder.ensure(BootstrapToml { profile, deny_warnings });
    }

    fn run(self, builder: &Builder<'_>) {
        setup(&builder.sess.config, self);
    }
}

pub fn setup(config: &Config, BootstrapToml { profile, deny_warnings }: BootstrapToml) {
    let suggestions: &[&str] = match profile {
        Profile::Compiler | Profile::None => &["check", "build", "test"],
        Profile::Tools => &[
            "check",
            "build",
            "test tests/rustdoc*",
            "test src/tools/clippy",
            "test src/tools/miri",
            "test src/tools/rustfmt",
        ],
        Profile::Library => &["check", "build", "test library/std", "doc"],
        Profile::Dist => &["dist", "build"],
    };

    println!();

    println!("To get started, try one of the following commands:");
    for cmd in suggestions {
        println!("- `x.py {cmd}`");
    }

    if profile != Profile::Dist {
        println!(
            "For more suggestions, see https://rustc-dev-guide.rust-lang.org/building/suggested.html"
        );
    }

    if profile == Profile::Tools {
        eprintln!();
        eprintln!(
            "NOTE: the `tools` profile sets up the `stage2` toolchain (use \
            `rustup toolchain link 'name' build/host/stage2` to use rustc)"
        )
    }

    let path = &config.config.clone().unwrap_or(PathBuf::from("bootstrap.toml"));
    setup_bootstrap_toml(path, profile, deny_warnings, config);
}

fn setup_bootstrap_toml(path: &Path, profile: Profile, deny_warnings: bool, config: &Config) {
    if profile == Profile::None {
        return;
    }

    let latest_change_id = CONFIG_CHANGE_HISTORY.last().unwrap().change_id;
    let rust_section = if deny_warnings { "" } else { "\n[rust]\ndeny-warnings = false\n" };

    let settings = format!(
        "# See bootstrap.example.toml for documentation of available options\n\
         #\n\
         profile = \"{profile}\"  # Includes one of the default files in {PROFILE_DIR}\n\
         change-id = {latest_change_id}\n\
         {rust_section}"
    );

    t!(fs::write(path, settings));

    let include_path = profile.include_path(&config.src);
    println!("`x.py` will now use the configuration at {}", include_path.display());
}

// Used to get the path for `Subcommand::Setup`
pub fn interactive_path() -> io::Result<Profile> {
    fn abbrev_all() -> impl Iterator<Item = ((String, String), Profile)> {
        ('a'..)
            .zip(1..)
            .map(|(letter, number)| (letter.to_string(), number.to_string()))
            .zip(Profile::all())
    }

    fn parse_with_abbrev(input: &str) -> Result<Profile, String> {
        let input = input.trim().to_lowercase();
        for ((letter, number), profile) in abbrev_all() {
            if input == letter || input == number {
                return Ok(profile);
            }
        }
        input.parse()
    }

    println!("Welcome to the Rust project! What do you want to do with x.py?");
    for ((letter, _), profile) in abbrev_all() {
        println!("{}) {}: {}", letter, profile, profile.purpose());
    }
    let template = loop {
        print!(
            "Please choose one ({}): ",
            abbrev_all().map(|((l, _), _)| l).collect::<Vec<_>>().join("/")
        );
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.is_empty() {
            eprintln!("EOF on stdin, when expecting answer to question.  Giving up.");
            helpers::exit_process(1);
        }
        break match parse_with_abbrev(&input) {
            Ok(profile) => profile,
            Err(err) => {
                eprintln!("ERROR: {err}");
                eprintln!("NOTE: press Ctrl+C to exit");
                continue;
            }
        };
    };
    Ok(template)
}
