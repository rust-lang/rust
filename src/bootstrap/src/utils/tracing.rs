//! Wrapper macros for `tracing` macros to avoid having to write `cfg(feature = "tracing")`-gated
//! `debug!`/`trace!` everytime, e.g.
//!
//! ```rust,ignore (example)
//! #[cfg(feature = "tracing")]
//! trace!("...");
//! ```
//!
//! When `feature = "tracing"` is inactive, these macros expand to nothing.

#[macro_export]
macro_rules! trace {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::trace!($($tokens)*)
    }
}

#[macro_export]
macro_rules! debug {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::debug!($($tokens)*)
    }
}

#[macro_export]
macro_rules! warn {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::warn!($($tokens)*)
    }
}

#[macro_export]
macro_rules! info {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::info!($($tokens)*)
    }
}

#[macro_export]
macro_rules! error {
    ($($tokens:tt)*) => {
        #[cfg(feature = "tracing")]
        ::tracing::error!($($tokens)*)
    }
}

#[cfg(feature = "tracing")]
pub const IO_SPAN_TARGET: &str = "IO";

/// Create a tracing span around an I/O operation, if tracing is enabled.
/// Note that at least one tracing value field has to be passed to this macro, otherwise it will not
/// compile.
#[macro_export]
macro_rules! trace_io {
    ($name:expr, $($args:tt)*) => {
        ::tracing::trace_span!(
            target: $crate::utils::tracing::IO_SPAN_TARGET,
            $name,
            $($args)*,
            location = $crate::utils::tracing::format_location(*::std::panic::Location::caller())
        ).entered()
    }
}

#[cfg(feature = "tracing")]
pub fn format_location(location: std::panic::Location<'static>) -> String {
    format!("{}:{}", location.file(), location.line())
}

#[cfg(feature = "tracing")]
const COMMAND_SPAN_TARGET: &str = "COMMAND";

#[cfg(feature = "tracing")]
pub fn trace_cmd(command: &crate::BootstrapCommand) -> tracing::span::EnteredSpan {
    let fingerprint = command.fingerprint();
    let location = command.get_created_location();
    let location = format_location(location);

    tracing::span!(
        target: COMMAND_SPAN_TARGET,
        tracing::Level::TRACE,
        "cmd",
        cmd_name = fingerprint.program_name().to_string(),
        cmd = fingerprint.format_short_cmd(),
        full_cmd = ?command,
        location
    )
    .entered()
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
mod inner {
    use std::fmt::Debug;
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    use chrono::{DateTime, Utc};
    use tracing::field::{Field, Visit};
    use tracing::{Event, Id, Level, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::registry::{LookupSpan, SpanRef};
    use tracing_subscriber::{EnvFilter, Layer};

    use super::{COMMAND_SPAN_TARGET, IO_SPAN_TARGET};
    use crate::STEP_SPAN_TARGET;

    pub fn setup_tracing(env_name: &str) -> TracingGuard {
        let filter = EnvFilter::from_env(env_name);

        let registry = tracing_subscriber::registry().with(filter).with(TracingPrinter::default());

        // When we're creating this layer, we do not yet know the location of the tracing output
        // directory, because it is stored in the output directory determined after Config is parsed,
        // but we already want to make tracing calls during (and before) config parsing.
        // So we store the output into a temporary file, and then move it to the tracing directory
        // before bootstrap ends.
        let tempdir = tempfile::TempDir::new().expect("Cannot create temporary directory");
        let chrome_tracing_path = tempdir.path().join("bootstrap-trace.json");
        let file = std::io::BufWriter::new(File::create(&chrome_tracing_path).unwrap());

        let chrome_layer = tracing_chrome::ChromeLayerBuilder::new()
            .writer(file)
            .include_args(true)
            .name_fn(Box::new(|event_or_span| match event_or_span {
                tracing_chrome::EventOrSpan::Event(e) => e.metadata().name().to_string(),
                tracing_chrome::EventOrSpan::Span(s) => {
                    if s.metadata().target() == STEP_SPAN_TARGET
                        && let Some(extension) = s.extensions().get::<StepNameExtension>()
                    {
                        extension.0.clone()
                    } else if s.metadata().target() == COMMAND_SPAN_TARGET
                        && let Some(extension) = s.extensions().get::<CommandNameExtension>()
                    {
                        extension.0.clone()
                    } else {
                        s.metadata().name().to_string()
                    }
                }
            }));
        let (chrome_layer, guard) = chrome_layer.build();

        tracing::subscriber::set_global_default(registry.with(chrome_layer)).unwrap();
        TracingGuard { guard, _tempdir: tempdir, chrome_tracing_path }
    }

    pub struct TracingGuard {
        guard: tracing_chrome::FlushGuard,
        _tempdir: tempfile::TempDir,
        chrome_tracing_path: std::path::PathBuf,
    }

    impl TracingGuard {
        pub fn copy_to_dir(self, dir: &std::path::Path) {
            drop(self.guard);
            crate::utils::helpers::move_file(
                &self.chrome_tracing_path,
                dir.join("chrome-trace.json"),
            )
            .unwrap();
        }
    }

    /// Visitor that extracts both known and unknown field values from events and spans.
    #[derive(Default)]
    struct FieldValues {
        /// Main event message
        message: Option<String>,
        /// Name of a recorded psna
        step_name: Option<String>,
        /// Short name of an executed command
        cmd_name: Option<String>,
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
                "cmd_name" => {
                    self.cmd_name = Some(value.to_string());
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

    /// Holds the name of a step span, stored in `tracing_subscriber`'s extensions.
    struct StepNameExtension(String);

    /// Holds the name of a command span, stored in `tracing_subscriber`'s extensions.
    struct CommandNameExtension(String);

    #[derive(Default)]
    struct TracingPrinter {
        indent: std::sync::atomic::AtomicU32,
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

            fn write_fields<'a, I: IntoIterator<Item = &'a (&'a str, String)>, W: Write>(
                writer: &mut W,
                iter: I,
            ) -> std::io::Result<()> {
                let items = iter.into_iter().collect::<Vec<_>>();
                if !items.is_empty() {
                    write!(writer, " [")?;
                    for (index, (name, value)) in items.iter().enumerate() {
                        write!(writer, "{name} = {value}")?;
                        if index < items.len() - 1 {
                            write!(writer, ", ")?;
                        }
                    }
                    write!(writer, "]")?;
                }
                Ok(())
            }

            // Write fields while treating the "location" field specially, and assuming that it
            // contains the source file location relevant to the span.
            let write_with_location = |writer: &mut W| -> std::io::Result<()> {
                if let Some(values) = field_values {
                    write_fields(
                        writer,
                        values.fields.iter().filter(|(name, _)| *name != "location"),
                    )?;
                    let location =
                        &values.fields.iter().find(|(name, _)| *name == "location").unwrap().1;
                    let (filename, line) = location.rsplit_once(':').unwrap();
                    let filename = shorten_filename(filename);
                    write!(writer, " ({filename}:{line})",)?;
                }
                Ok(())
            };

            // We handle steps specially. We instrument them dynamically in `Builder::ensure`,
            // and we want to have custom name for each step span. But tracing doesn't allow setting
            // dynamic span names. So we detect step spans here and override their name.
            match span.metadata().target() {
                // Executed step
                STEP_SPAN_TARGET => {
                    let name =
                        field_values.and_then(|v| v.step_name.as_deref()).unwrap_or(span.name());
                    write!(writer, "{name}")?;

                    // There should be only one more field called `args`
                    if let Some(values) = field_values {
                        let field = &values.fields[0];
                        write!(writer, " {{{}}}", field.1)?;
                    }
                    write_location(writer, span.metadata())?;
                }
                // Executed command
                COMMAND_SPAN_TARGET => {
                    write!(writer, "{}", span.name())?;
                    write_with_location(writer)?;
                }
                IO_SPAN_TARGET => {
                    write!(writer, "{}", span.name())?;
                    write_with_location(writer)?;
                }
                // Other span
                _ => {
                    write!(writer, "{}", span.name())?;
                    if let Some(values) = field_values {
                        write_fields(writer, values.fields.iter())?;
                    }
                    write_location(writer, span.metadata())?;
                }
            }

            writeln!(writer)?;
            Ok(())
        }
    }

    fn write_location<W: Write>(
        writer: &mut W,
        metadata: &'static tracing::Metadata<'static>,
    ) -> std::io::Result<()> {
        if let Some(filename) = metadata.file() {
            let filename = shorten_filename(filename);

            write!(writer, " ({filename}")?;
            if let Some(line) = metadata.line() {
                write!(writer, ":{line}")?;
            }
            write!(writer, ")")?;
        }
        Ok(())
    }

    /// Keep only the module name and file name to make it shorter
    fn shorten_filename(filename: &str) -> String {
        Path::new(filename)
            .components()
            // Take last two path components
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<PathBuf>()
            .display()
            .to_string()
    }

    impl<S> Layer<S> for TracingPrinter
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            // Record value of span fields
            // Note that we do not implement changing values of span fields after they are created.
            // For that we would also need to implement the `on_record` method
            let mut field_values = FieldValues::default();
            attrs.record(&mut field_values);

            // We need to propagate the actual name of the span to the Chrome layer below, because
            // it cannot access field values. We do that through extensions.
            if attrs.metadata().target() == STEP_SPAN_TARGET
                && let Some(step_name) = field_values.step_name.clone()
            {
                ctx.span(id).unwrap().extensions_mut().insert(StepNameExtension(step_name));
            } else if attrs.metadata().target() == COMMAND_SPAN_TARGET
                && let Some(cmd_name) = field_values.cmd_name.clone()
            {
                ctx.span(id).unwrap().extensions_mut().insert(CommandNameExtension(cmd_name));
            }
            self.span_values.lock().unwrap().insert(id.clone(), field_values);
        }

        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut writer = std::io::stderr().lock();
            self.write_event(&mut writer, event).unwrap();
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
}

#[cfg(feature = "tracing")]
pub use inner::setup_tracing;
