//! bootstrap, the Rust build system
//!
//! This is the entry point for the build system used to compile the `rustc`
//! compiler. Lots of documentation can be found in the `README.md` file in the
//! parent directory, and otherwise documentation can be found throughout the `build`
//! directory in each respective module.

use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::str::FromStr;
use std::time::Instant;
use std::{env, process};

use bootstrap::{
    Build, CONFIG_CHANGE_HISTORY, ChangeId, Config, Flags, Subcommand, debug,
    find_recent_config_change_ids, human_readable_changes, symlink_dir, t,
};
#[cfg(feature = "tracing")]
use tracing::instrument;

fn is_profiling_enabled() -> bool {
    env::var("BOOTSTRAP_PROFILE").is_ok_and(|v| v == "1")
}

fn is_tracing_enabled() -> bool {
    is_profiling_enabled() || cfg!(feature = "tracing")
}

#[cfg_attr(feature = "tracing", instrument(level = "trace", name = "main"))]
fn main() {
    #[cfg(feature = "tracing")]
    let guard = setup_tracing(is_profiling_enabled());

    let start_time = Instant::now();

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

    let tracing_enabled = is_tracing_enabled();

    // Prepare a directory for tracing output
    // Also store a symlink named "latest" to point to the latest tracing directory.
    let tracing_dir = out_dir.join("bootstrap-trace").join(std::process::id().to_string());
    let latest_trace_dir = tracing_dir.parent().unwrap().join("latest");
    if tracing_enabled {
        let _ = std::fs::remove_dir_all(&tracing_dir);
        std::fs::create_dir_all(&tracing_dir).unwrap();

        #[cfg(windows)]
        let _ = std::fs::remove_dir(&latest_trace_dir);
        #[cfg(not(windows))]
        let _ = std::fs::remove_file(&latest_trace_dir);

        t!(symlink_dir(&config, &tracing_dir, &latest_trace_dir));
    }

    debug!("creating new build based on config");
    let mut build = Build::new(config);
    build.build();

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

    if is_profiling_enabled() {
        build.report_summary(&tracing_dir.join("command-stats.txt"), start_time);
    }

    #[cfg(feature = "tracing")]
    {
        build.report_step_graph(&tracing_dir);
        if let Some(guard) = guard {
            guard.copy_to_dir(&tracing_dir);
        }
    }

    if tracing_enabled {
        eprintln!("Tracing/profiling output has been written to {}", latest_trace_dir.display());
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
fn setup_tracing(profiling_enabled: bool) -> Option<TracingGuard> {
    use std::fmt::Debug;
    use std::fs::File;
    use std::io::BufWriter;
    use std::sync::atomic::{AtomicU32, Ordering};

    use bootstrap::STEP_NAME_TARGET;
    use chrono::{DateTime, Utc};
    use tracing::field::{Field, Visit};
    use tracing::{Event, Id, Level, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::registry::{LookupSpan, SpanRef};
    use tracing_subscriber::{EnvFilter, Layer};

    let filter = EnvFilter::from_env("BOOTSTRAP_TRACING");

    /// Visitor that extracts both known and unknown field values from events and spans.
    #[derive(Default)]
    struct FieldValues {
        /// Main event message
        message: Option<String>,
        /// Name of a recorded psna
        step_name: Option<String>,
        /// The rest of arbitrary event/span fields
        fields: Vec<(&'static str, String)>,
    }

    impl Visit for FieldValues {
        /// Record fields if possible using `record_str`, to avoid rendering simple strings with
        /// their `Debug` representation, which adds extra quotes.
        fn record_str(&mut self, field: &Field, value: &str) {
            match field.name() {
                "step_name" => {
                    self.step_name = Some(value.to_string());
                }
                name => {
                    self.fields.push((name, value.to_string()));
                }
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
            let formatted = format!("{value:?}");
            match field.name() {
                "message" => {
                    self.message = Some(formatted);
                }
                name => {
                    self.fields.push((name, formatted));
                }
            }
        }
    }

    #[derive(Copy, Clone)]
    enum SpanAction {
        Enter,
    }

    /// Holds the name of a step, stored in `tracing_subscriber`'s extensions.
    struct StepNameExtension(String);

    #[derive(Default)]
    struct TracingPrinter {
        indent: AtomicU32,
        span_values: std::sync::Mutex<std::collections::HashMap<tracing::Id, FieldValues>>,
    }

    impl TracingPrinter {
        fn format_header<W: Write>(
            &self,
            writer: &mut W,
            time: DateTime<Utc>,
            level: &Level,
        ) -> std::io::Result<()> {
            // Use a fixed-width timestamp without date, that shouldn't be very important
            let timestamp = time.format("%H:%M:%S.%3f");
            write!(writer, "{timestamp} ")?;
            // Make sure that levels are aligned to the same number of characters, in order not to
            // break the layout
            write!(writer, "{level:>5} ")?;
            write!(writer, "{}", " ".repeat(self.indent.load(Ordering::Relaxed) as usize))
        }

        fn write_event<W: Write>(&self, writer: &mut W, event: &Event<'_>) -> std::io::Result<()> {
            let now = Utc::now();

            self.format_header(writer, now, event.metadata().level())?;

            let mut field_values = FieldValues::default();
            event.record(&mut field_values);

            if let Some(msg) = &field_values.message {
                write!(writer, "{msg}")?;
            }

            if !field_values.fields.is_empty() {
                if field_values.message.is_some() {
                    write!(writer, " ")?;
                }
                write!(writer, "[")?;
                for (index, (name, value)) in field_values.fields.iter().enumerate() {
                    write!(writer, "{name} = {value}")?;
                    if index < field_values.fields.len() - 1 {
                        write!(writer, ", ")?;
                    }
                }
                write!(writer, "]")?;
            }
            write_location(writer, event.metadata())?;
            writeln!(writer)?;
            Ok(())
        }

        fn write_span<W: Write, S>(
            &self,
            writer: &mut W,
            span: SpanRef<'_, S>,
            field_values: Option<&FieldValues>,
            action: SpanAction,
        ) -> std::io::Result<()>
        where
            S: for<'lookup> LookupSpan<'lookup>,
        {
            let now = Utc::now();

            self.format_header(writer, now, span.metadata().level())?;
            match action {
                SpanAction::Enter => {
                    write!(writer, "> ")?;
                }
            }

            // We handle steps specially. We instrument them dynamically in `Builder::ensure`,
            // and we want to have custom name for each step span. But tracing doesn't allow setting
            // dynamic span names. So we detect step spans here and override their name.
            if span.metadata().target() == STEP_NAME_TARGET {
                let name = field_values.and_then(|v| v.step_name.as_deref()).unwrap_or(span.name());
                write!(writer, "{name}")?;

                // There should be only one more field called `args`
                if let Some(values) = field_values {
                    let field = &values.fields[0];
                    write!(writer, " {{{}}}", field.1)?;
                }
            } else {
                write!(writer, "{}", span.name())?;
                if let Some(values) = field_values.filter(|v| !v.fields.is_empty()) {
                    write!(writer, " [")?;
                    for (index, (name, value)) in values.fields.iter().enumerate() {
                        write!(writer, "{name} = {value}")?;
                        if index < values.fields.len() - 1 {
                            write!(writer, ", ")?;
                        }
                    }
                    write!(writer, "]")?;
                }
            };

            write_location(writer, span.metadata())?;
            writeln!(writer)?;
            Ok(())
        }
    }

    fn write_location<W: Write>(
        writer: &mut W,
        metadata: &'static tracing::Metadata<'static>,
    ) -> std::io::Result<()> {
        use std::path::{Path, PathBuf};

        if let Some(filename) = metadata.file() {
            // Keep only the module name and file name to make it shorter
            let filename: PathBuf = Path::new(filename)
                .components()
                // Take last two path components
                .rev()
                .take(2)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();

            write!(writer, " ({}", filename.display())?;
            if let Some(line) = metadata.line() {
                write!(writer, ":{line}")?;
            }
            write!(writer, ")")?;
        }
        Ok(())
    }

    impl<S> Layer<S> for TracingPrinter
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut writer = std::io::stderr().lock();
            self.write_event(&mut writer, event).unwrap();
        }

        fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            // Record value of span fields
            // Note that we do not implement changing values of span fields after they are created.
            // For that we would also need to implement the `on_record` method
            let mut field_values = FieldValues::default();
            attrs.record(&mut field_values);

            // We need to propagate the actual name of the span to the Chrome layer below, because
            // it cannot access field values. We do that through extensions.
            if let Some(step_name) = field_values.step_name.clone() {
                ctx.span(id).unwrap().extensions_mut().insert(StepNameExtension(step_name));
            }
            self.span_values.lock().unwrap().insert(id.clone(), field_values);
        }

        fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                let mut writer = std::io::stderr().lock();
                let values = self.span_values.lock().unwrap();
                let values = values.get(id);
                self.write_span(&mut writer, span, values, SpanAction::Enter).unwrap();
            }
            self.indent.fetch_add(1, Ordering::Relaxed);
        }

        fn on_exit(&self, _id: &Id, _ctx: Context<'_, S>) {
            self.indent.fetch_sub(1, Ordering::Relaxed);
        }
    }

    let registry = tracing_subscriber::registry().with(filter).with(TracingPrinter::default());

    let guard = if profiling_enabled {
        // When we're creating this layer, we do not yet know the location of the tracing output
        // directory, because it is stored in the output directory determined after Config is parsed,
        // but we already want to make tracing calls during (and before) config parsing.
        // So we store the output into a temporary file, and then move it to the tracing directory
        // before bootstrap ends.
        let tempdir = tempfile::TempDir::new().expect("Cannot create temporary directory");
        let chrome_tracing_path = tempdir.path().join("bootstrap-trace.json");
        let file = BufWriter::new(File::create(&chrome_tracing_path).unwrap());

        let chrome_layer = tracing_chrome::ChromeLayerBuilder::new()
            .writer(file)
            .include_args(true)
            .name_fn(Box::new(|event_or_span| match event_or_span {
                tracing_chrome::EventOrSpan::Event(e) => e.metadata().name().to_string(),
                tracing_chrome::EventOrSpan::Span(s) => {
                    if s.metadata().target() == STEP_NAME_TARGET
                        && let Some(extension) = s.extensions().get::<StepNameExtension>()
                    {
                        extension.0.clone()
                    } else {
                        s.metadata().name().to_string()
                    }
                }
            }));
        let (chrome_layer, guard) = chrome_layer.build();

        tracing::subscriber::set_global_default(registry.with(chrome_layer)).unwrap();
        Some(TracingGuard { guard, _tempdir: tempdir, chrome_tracing_path })
    } else {
        tracing::subscriber::set_global_default(registry).unwrap();
        None
    };

    guard
}

#[cfg(feature = "tracing")]
struct TracingGuard {
    guard: tracing_chrome::FlushGuard,
    _tempdir: tempfile::TempDir,
    chrome_tracing_path: std::path::PathBuf,
}

#[cfg(feature = "tracing")]
impl TracingGuard {
    fn copy_to_dir(self, dir: &std::path::Path) {
        drop(self.guard);
        std::fs::rename(&self.chrome_tracing_path, dir.join("chrome-trace.json")).unwrap();
    }
}
