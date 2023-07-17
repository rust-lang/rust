//! Simple logger that logs either to stderr or to a file, using `tracing_subscriber`
//! filter syntax and `tracing_appender` for non blocking output.

use std::{
    fmt,
    fs::File,
    io::{self, Stderr},
    sync::Arc,
};

use anyhow::Context;
use tracing::{level_filters::LevelFilter, Event, Subscriber};
use tracing_log::NormalizeEvent;
use tracing_subscriber::{
    filter::Targets,
    fmt::{
        format::Writer, writer::BoxMakeWriter, FmtContext, FormatEvent, FormatFields,
        FormattedFields, MakeWriter,
    },
    layer::SubscriberExt,
    registry::LookupSpan,
    util::SubscriberInitExt,
    Registry,
};
use tracing_tree::HierarchicalLayer;

pub(crate) struct LoggerConfig {
    pub(crate) log_file: Option<File>,
    pub(crate) filter: String,
    pub(crate) chalk_filter: Option<String>,
}

struct MakeWriterStderr;

impl MakeWriter<'_> for MakeWriterStderr {
    type Writer = Stderr;

    fn make_writer(&self) -> Self::Writer {
        io::stderr()
    }
}

impl LoggerConfig {
    pub(crate) fn init(self) -> anyhow::Result<()> {
        let mut filter: Targets = self
            .filter
            .parse()
            .with_context(|| format!("invalid log filter: `{}`", self.filter))?;

        let mut chalk_layer = None;
        if let Some(chalk_filter) = self.chalk_filter {
            let level: LevelFilter =
                chalk_filter.parse().with_context(|| "invalid chalk log filter")?;
            chalk_layer = Some(
                HierarchicalLayer::default()
                    .with_indent_lines(true)
                    .with_ansi(false)
                    .with_indent_amount(2)
                    .with_writer(io::stderr),
            );
            filter = filter
                .with_target("chalk_solve", level)
                .with_target("chalk_ir", level)
                .with_target("chalk_recursive", level);
        };

        let writer = match self.log_file {
            Some(file) => BoxMakeWriter::new(Arc::new(file)),
            None => BoxMakeWriter::new(io::stderr),
        };
        let ra_fmt_layer =
            tracing_subscriber::fmt::layer().event_format(LoggerFormatter).with_writer(writer);

        let registry = Registry::default().with(filter).with(ra_fmt_layer);
        match chalk_layer {
            Some(chalk_layer) => registry.with(chalk_layer).init(),
            None => registry.init(),
        }
        Ok(())
    }
}

#[derive(Debug)]
struct LoggerFormatter;

impl<S, N> FormatEvent<S, N> for LoggerFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // Write level and target
        let level = *event.metadata().level();

        // If this event is issued from `log` crate, then the value of target is
        // always "log". `tracing-log` has hard coded it for some reason, so we
        // need to extract it using `normalized_metadata` method which is part of
        // `tracing_log::NormalizeEvent`.
        let target = match event.normalized_metadata() {
            // This event is issued from `log` crate
            Some(log) => log.target(),
            None => event.metadata().target(),
        };
        write!(writer, "[{level} {target}] ")?;

        // Write spans and fields of each span
        ctx.visit_spans(|span| {
            write!(writer, "{}", span.name())?;

            let ext = span.extensions();

            // `FormattedFields` is a formatted representation of the span's
            // fields, which is stored in its extensions by the `fmt` layer's
            // `new_span` method. The fields will have been formatted
            // by the same field formatter that's provided to the event
            // formatter in the `FmtContext`.
            let fields = &ext.get::<FormattedFields<N>>().expect("will never be `None`");

            if !fields.is_empty() {
                write!(writer, "{{{fields}}}")?;
            }
            write!(writer, ": ")?;

            Ok(())
        })?;

        // Write fields on the event
        ctx.field_format().format_fields(writer.by_ref(), event)?;

        writeln!(writer)
    }
}
