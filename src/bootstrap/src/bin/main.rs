//! bootstrap, the Rust build system
//!
//! This is the entry point for the build system used to compile the `rustc`
//! compiler. Lots of documentation can be found in the `README.md` file in the
//! parent directory, and otherwise documentation can be found throughout the `build`
//! directory in each respective module.

use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::str::FromStr;
use std::{env, process};

use bootstrap::{
    Build, CONFIG_CHANGE_HISTORY, ChangeId, Config, Flags, Subcommand, debug,
    find_recent_config_change_ids, human_readable_changes, t,
};
#[cfg(feature = "tracing")]
use tracing::instrument;

#[cfg_attr(feature = "tracing", instrument(level = "trace", name = "main"))]
fn main() {
    #[cfg(feature = "tracing")]
    let _guard = setup_tracing();

    let args = env::args().skip(1).collect::<Vec<_>>();

    if Flags::try_parse_verbose_help(&args) {
        return;
    }

    debug!("parsing flags");
    let flags = Flags::parse(&args);
    debug!("parsing config based on flags");
    let config = Config::parse(flags);

    let mut build_lock;
    let _build_lock_guard;

    if !config.bypass_bootstrap_lock {
        // Display PID of process holding the lock
        // PID will be stored in a lock file
        let lock_path = config.out.join("lock");
        let pid = fs::read_to_string(&lock_path);

        build_lock = fd_lock::RwLock::new(t!(fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&lock_path)));
        _build_lock_guard = match build_lock.try_write() {
            Ok(mut lock) => {
                t!(lock.write(process::id().to_string().as_ref()));
                lock
            }
            err => {
                drop(err);
                // #135972: We can reach this point when the lock has been taken,
                // but the locker has not yet written its PID to the file
                if let Some(pid) = pid.ok().filter(|pid| !pid.is_empty()) {
                    println!("WARNING: build directory locked by process {pid}, waiting for lock");
                } else {
                    println!("WARNING: build directory locked, waiting for lock");
                }
                let mut lock = t!(build_lock.write());
                t!(lock.write(process::id().to_string().as_ref()));
                lock
            }
        };
    }

    // check_version warnings are not printed during setup, or during CI
    let changelog_suggestion = if matches!(config.cmd, Subcommand::Setup { .. })
        || config.is_running_on_ci
        || config.dry_run()
    {
        None
    } else {
        check_version(&config)
    };

    // NOTE: Since `./configure` generates a `bootstrap.toml`, distro maintainers will see the
    // changelog warning, not the `x.py setup` message.
    let suggest_setup = config.config.is_none() && !matches!(config.cmd, Subcommand::Setup { .. });
    if suggest_setup {
        println!("WARNING: you have not made a `bootstrap.toml`");
        println!(
            "HELP: consider running `./x.py setup` or copying `bootstrap.example.toml` by running \
            `cp bootstrap.example.toml bootstrap.toml`"
        );
    } else if let Some(suggestion) = &changelog_suggestion {
        println!("{suggestion}");
    }

    let pre_commit = config.src.join(".git").join("hooks").join("pre-commit");
    let dump_bootstrap_shims = config.dump_bootstrap_shims;
    let out_dir = config.out.clone();

    debug!("creating new build based on config");
    Build::new(config).build();

    if suggest_setup {
        println!("WARNING: you have not made a `bootstrap.toml`");
        println!(
            "HELP: consider running `./x.py setup` or copying `bootstrap.example.toml` by running \
            `cp bootstrap.example.toml bootstrap.toml`"
        );
    } else if let Some(suggestion) = &changelog_suggestion {
        println!("{suggestion}");
    }

    // Give a warning if the pre-commit script is in pre-commit and not pre-push.
    // HACK: Since the commit script uses hard links, we can't actually tell if it was installed by x.py setup or not.
    // We could see if it's identical to src/etc/pre-push.sh, but pre-push may have been modified in the meantime.
    // Instead, look for this comment, which is almost certainly not in any custom hook.
    if fs::read_to_string(pre_commit).is_ok_and(|contents| {
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

    if dump_bootstrap_shims {
        let dump_dir = out_dir.join("bootstrap-shims-dump");
        assert!(dump_dir.exists());

        for entry in walkdir::WalkDir::new(&dump_dir) {
            let entry = t!(entry);

            if !entry.file_type().is_file() {
                continue;
            }

            let file = t!(fs::File::open(entry.path()));

            // To ensure deterministic results we must sort the dump lines.
            // This is necessary because the order of rustc invocations different
            // almost all the time.
            let mut lines: Vec<String> = t!(BufReader::new(&file).lines().collect());
            lines.sort_by_key(|t| t.to_lowercase());
            let mut file = t!(OpenOptions::new().write(true).truncate(true).open(entry.path()));
            t!(file.write_all(lines.join("\n").as_bytes()));
        }
    }
}

fn check_version(config: &Config) -> Option<String> {
    let mut msg = String::new();

    let latest_change_id = CONFIG_CHANGE_HISTORY.last().unwrap().change_id;
    let warned_id_path = config.out.join("bootstrap").join(".last-warned-change-id");

    let mut id = match config.change_id {
        Some(ChangeId::Id(id)) if id == latest_change_id => return None,
        Some(ChangeId::Ignore) => return None,
        Some(ChangeId::Id(id)) => id,
        None => {
            msg.push_str("WARNING: The `change-id` is missing in the `bootstrap.toml`. This means that you will not be able to track the major changes made to the bootstrap configurations.\n");
            msg.push_str("NOTE: to silence this warning, ");
            msg.push_str(&format!(
                "add `change-id = {latest_change_id}` or `change-id = \"ignore\"` at the top of `bootstrap.toml`"
            ));
            return Some(msg);
        }
    };

    // Always try to use `change-id` from .last-warned-change-id first. If it doesn't exist,
    // then use the one from the bootstrap.toml. This way we never show the same warnings
    // more than once.
    if let Ok(t) = fs::read_to_string(&warned_id_path) {
        let last_warned_id = usize::from_str(&t)
            .unwrap_or_else(|_| panic!("{} is corrupted.", warned_id_path.display()));

        // We only use the last_warned_id if it exists in `CONFIG_CHANGE_HISTORY`.
        // Otherwise, we may retrieve all the changes if it's not the highest value.
        // For better understanding, refer to `change_tracker::find_recent_config_change_ids`.
        if CONFIG_CHANGE_HISTORY.iter().any(|config| config.change_id == last_warned_id) {
            id = last_warned_id;
        }
    };

    let changes = find_recent_config_change_ids(id);

    if changes.is_empty() {
        return None;
    }

    msg.push_str("There have been changes to x.py since you last updated:\n");
    msg.push_str(&human_readable_changes(changes));

    msg.push_str("NOTE: to silence this warning, ");
    msg.push_str(&format!(
        "update `bootstrap.toml` to use `change-id = {latest_change_id}` or `change-id = \"ignore\"` instead"
    ));

    if io::stdout().is_terminal() {
        t!(fs::write(warned_id_path, latest_change_id.to_string()));
    }

    Some(msg)
}

// # Note on `tracing` usage in bootstrap
//
// Due to the conditional compilation via the `tracing` cargo feature, this means that `tracing`
// usages in bootstrap need to be also gated behind the `tracing` feature:
//
// - `tracing` macros with log levels (`trace!`, `debug!`, `warn!`, `info`, `error`) should not be
//   used *directly*. You should use the wrapped `tracing` macros which gate the actual invocations
//   behind `feature = "tracing"`.
// - `tracing`'s `#[instrument(..)]` macro will need to be gated like `#![cfg_attr(feature =
//   "tracing", instrument(..))]`.
#[cfg(feature = "tracing")]
fn setup_tracing() -> impl Drop {
    use tracing_forest::ForestLayer;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;

    let filter = EnvFilter::from_env("BOOTSTRAP_TRACING");

    let mut chrome_layer = tracing_chrome::ChromeLayerBuilder::new().include_args(true);

    // Writes the Chrome profile to trace-<unix-timestamp>.json if enabled
    if !env::var("BOOTSTRAP_PROFILE").is_ok_and(|v| v == "1") {
        chrome_layer = chrome_layer.writer(io::sink());
    }

    let (chrome_layer, _guard) = chrome_layer.build();

    let registry =
        tracing_subscriber::registry().with(filter).with(ForestLayer::default()).with(chrome_layer);

    tracing::subscriber::set_global_default(registry).unwrap();
    _guard
}
