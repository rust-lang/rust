//! rustbuild, the Rust build system
//!
//! This is the entry point for the build system used to compile the `rustc`
//! compiler. Lots of documentation can be found in the `README.md` file in the
//! parent directory, and otherwise documentation can be found throughout the `build`
//! directory in each respective module.

#[cfg(all(any(unix, windows), not(target_os = "solaris")))]
use std::io::Write;
#[cfg(all(any(unix, windows), not(target_os = "solaris")))]
use std::process;
use std::{env, fs};

#[cfg(all(any(unix, windows), not(target_os = "solaris")))]
use bootstrap::t;
use bootstrap::{find_recent_config_change_ids, Build, Config, Subcommand, CONFIG_CHANGE_HISTORY};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let config = Config::parse(&args);

    #[cfg(all(any(unix, windows), not(target_os = "solaris")))]
    let mut build_lock;
    #[cfg(all(any(unix, windows), not(target_os = "solaris")))]
    let _build_lock_guard;
    #[cfg(all(any(unix, windows), not(target_os = "solaris")))]
    // Display PID of process holding the lock
    // PID will be stored in a lock file
    {
        let path = config.out.join("lock");
        let pid = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(_) => String::new(),
        };

        build_lock =
            fd_lock::RwLock::new(t!(fs::OpenOptions::new().write(true).create(true).open(&path)));
        _build_lock_guard = match build_lock.try_write() {
            Ok(mut lock) => {
                t!(lock.write(&process::id().to_string().as_ref()));
                lock
            }
            err => {
                drop(err);
                println!("WARNING: build directory locked by process {pid}, waiting for lock");
                let mut lock = t!(build_lock.write());
                t!(lock.write(&process::id().to_string().as_ref()));
                lock
            }
        };
    }

    #[cfg(any(not(any(unix, windows)), target_os = "solaris"))]
    println!("WARNING: file locking not supported for target, not locking build directory");

    // check_version warnings are not printed during setup
    let changelog_suggestion =
        if matches!(config.cmd, Subcommand::Setup { .. }) { None } else { check_version(&config) };

    // NOTE: Since `./configure` generates a `config.toml`, distro maintainers will see the
    // changelog warning, not the `x.py setup` message.
    let suggest_setup = config.config.is_none() && !matches!(config.cmd, Subcommand::Setup { .. });
    if suggest_setup {
        println!("WARNING: you have not made a `config.toml`");
        println!(
            "HELP: consider running `./x.py setup` or copying `config.example.toml` by running \
            `cp config.example.toml config.toml`"
        );
    } else if let Some(suggestion) = &changelog_suggestion {
        println!("{suggestion}");
    }

    let pre_commit = config.src.join(".git").join("hooks").join("pre-commit");
    Build::new(config).build();

    if suggest_setup {
        println!("WARNING: you have not made a `config.toml`");
        println!(
            "HELP: consider running `./x.py setup` or copying `config.example.toml` by running \
            `cp config.example.toml config.toml`"
        );
    } else if let Some(suggestion) = &changelog_suggestion {
        println!("{suggestion}");
    }

    // Give a warning if the pre-commit script is in pre-commit and not pre-push.
    // HACK: Since the commit script uses hard links, we can't actually tell if it was installed by x.py setup or not.
    // We could see if it's identical to src/etc/pre-push.sh, but pre-push may have been modified in the meantime.
    // Instead, look for this comment, which is almost certainly not in any custom hook.
    if fs::read_to_string(pre_commit).map_or(false, |contents| {
        contents.contains("https://github.com/rust-lang/rust/issues/77620#issuecomment-705144570")
    }) {
        println!(
            "WARNING: You have the pre-push script installed to .git/hooks/pre-commit. \
                  Consider moving it to .git/hooks/pre-push instead, which runs less often."
        );
    }

    if suggest_setup || changelog_suggestion.is_some() {
        println!("NOTE: this message was printed twice to make it more likely to be seen");
    }
}

fn check_version(config: &Config) -> Option<String> {
    let mut msg = String::new();

    if config.changelog_seen.is_some() {
        msg.push_str("WARNING: The use of `changelog-seen` is deprecated. Please refer to `change-id` option in `config.example.toml` instead.\n");
    }

    let latest_config_id = CONFIG_CHANGE_HISTORY.last().unwrap();
    if let Some(id) = config.change_id {
        if &id == latest_config_id {
            return None;
        }

        let change_links: Vec<String> = find_recent_config_change_ids(id)
            .iter()
            .map(|id| format!("https://github.com/rust-lang/rust/pull/{id}"))
            .collect();
        if !change_links.is_empty() {
            msg.push_str("WARNING: there have been changes to x.py since you last updated.\n");
            msg.push_str("To see more detail about these changes, visit the following PRs:\n");

            for link in change_links {
                msg.push_str(&format!("  - {link}\n"));
            }

            msg.push_str("WARNING: there have been changes to x.py since you last updated.\n");

            msg.push_str("NOTE: to silence this warning, ");
            msg.push_str(&format!(
                "update `config.toml` to use `change-id = {latest_config_id}` instead"
            ));
        }
    } else {
        msg.push_str("WARNING: The `change-id` is missing in the `config.toml`. This means that you will not be able to track the major changes made to the bootstrap configurations.\n");
        msg.push_str("NOTE: to silence this warning, ");
        msg.push_str(&format!("add `change-id = {latest_config_id}` at the top of `config.toml`"));
    };

    Some(msg)
}
