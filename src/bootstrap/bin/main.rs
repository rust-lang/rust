//! rustbuild, the Rust build system
//!
//! This is the entry point for the build system used to compile the `rustc`
//! compiler. Lots of documentation can be found in the `README.md` file in the
//! parent directory, and otherwise documentation can be found throughout the `build`
//! directory in each respective module.

use std::env;

use bootstrap::{Build, Config, Subcommand, VERSION};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let config = Config::parse(&args);

    // check_version warnings are not printed during setup
    let changelog_suggestion =
        if matches!(config.cmd, Subcommand::Setup {..}) { None } else { check_version(&config) };

    // NOTE: Since `./configure` generates a `config.toml`, distro maintainers will see the
    // changelog warning, not the `x.py setup` message.
    let suggest_setup = !config.config.exists() && !matches!(config.cmd, Subcommand::Setup { .. });
    if suggest_setup {
        println!("warning: you have not made a `config.toml`");
        println!("help: consider running `x.py setup` or copying `config.toml.example`");
    } else if let Some(suggestion) = &changelog_suggestion {
        println!("{}", suggestion);
    }

    Build::new(config).build();

    if suggest_setup {
        println!("warning: you have not made a `config.toml`");
        println!("help: consider running `x.py setup` or copying `config.toml.example`");
    } else if let Some(suggestion) = &changelog_suggestion {
        println!("{}", suggestion);
    }

    if suggest_setup || changelog_suggestion.is_some() {
        println!("note: this message was printed twice to make it more likely to be seen");
    }
}

fn check_version(config: &Config) -> Option<String> {
    let mut msg = String::new();

    let suggestion = if let Some(seen) = config.changelog_seen {
        if seen != VERSION {
            msg.push_str("warning: there have been changes to x.py since you last updated.\n");
            format!("update `config.toml` to use `changelog-seen = {}` instead", VERSION)
        } else {
            return None;
        }
    } else {
        msg.push_str("warning: x.py has made several changes recently you may want to look at\n");
        format!("add `changelog-seen = {}` at the top of `config.toml`", VERSION)
    };

    msg.push_str("help: consider looking at the changes in `src/bootstrap/CHANGELOG.md`\n");
    msg.push_str("note: to silence this warning, ");
    msg.push_str(&suggestion);

    Some(msg)
}
