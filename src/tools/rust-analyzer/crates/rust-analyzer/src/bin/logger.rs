//! Simple logger that logs either to stderr or to a file, using `tracing_subscriber`
//! filter syntax and `tracing_appender` for non blocking output.

use std::{
    fmt,
    fs::File,
    io::{self, Stderr},
    sync::Arc,
};

use rust_analyzer::Result;
use tracing::{level_filters::LevelFilter, Event, Subscriber};
use tracing_log::NormalizeEvent;
use tracing_subscriber::{
    fmt::{
        format::Writer, writer::BoxMakeWriter, FmtContext, FormatEvent, FormatFields,
        FormattedFields, MakeWriter,
    },
    layer::SubscriberExt,
    registry::LookupSpan,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};
use tracing_tree::HierarchicalLayer;

pub(crate) struct Logger {
    filter: EnvFilter,
    file: Option<File>,
}

struct MakeWriterStderr;

impl<'a> MakeWriter<'a> for MakeWriterStderr {
    type Writer = Stderr;

    fn make_writer(&'a self) -> Self::Writer {
        io::stderr()
    }
}

impl Logger {
    pub(crate) fn new(file: Option<File>, filter: Option<&str>) -> Logger {
        let filter = filter.map_or(EnvFilter::default(), EnvFilter::new);

        Logger { filter, file }
    }

    pub(crate) fn install(self) -> Result<()> {
        // The meaning of CHALK_DEBUG I suspected is to tell chalk crates
        // (i.e. chalk-solve, chalk-ir, chalk-recursive) how to filter tracing
        // logs. But now we can only have just one filter, which means we have to
        // merge chalk filter to our main filter (from RA_LOG env).
        //
        // The acceptable syntax of CHALK_DEBUG is `target[span{field=value}]=level`.
        // As the value should only affect chalk crates, we'd better manually
        // specify the target. And for simplicity, CHALK_DEBUG only accept the value
        // that specify level.
        let chalk_level_dir = std::env::var("CHALK_DEBUG")
            .map(|val| {
                val.parse::<LevelFilter>().expect(
                    "invalid CHALK_DEBUG value, expect right log level (like debug or trace)",
                )
            })
            .ok();

        let chalk_layer = HierarchicalLayer::default()
            .with_indent_lines(true)
            .with_ansi(false)
            .with_indent_amount(2)
            .with_writer(io::stderr);

        let writer = match self.file {
            Some(file) => BoxMakeWriter::new(Arc::new(file)),
            None => BoxMakeWriter::new(io::stderr),
        };
        let ra_fmt_layer =
            tracing_subscriber::fmt::layer().event_format(LoggerFormatter).with_writer(writer);

        match chalk_level_dir {
            Some(val) => {
                Registry::default()
                    .with(
                        self.filter
                            .add_directive(format!("chalk_solve={}", val).parse()?)
                            .add_directive(format!("chalk_ir={}", val).parse()?)
                            .add_directive(format!("chalk_recursive={}", val).parse()?),
                    )
                    .with(ra_fmt_layer)
                    .with(chalk_layer)
                    .init();
            }
            None => {
                Registry::default().with(self.filter).with(ra_fmt_layer).init();
            }
        };

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
        write!(writer, "[{} {}] ", level, target)?;

        // Write spans and fields of each span
        ctx.visit_spans(|span| {
            write!(writer, "{}", span.name())?;

            let ext = span.extensions();

            // `FormattedFields` is a a formatted representation of the span's
            // fields, which is stored in its extensions by the `fmt` layer's
            // `new_span` method. The fields will have been formatted
            // by the same field formatter that's provided to the event
            // formatter in the `FmtContext`.
            let fields = &ext.get::<FormattedFields<N>>().expect("will never be `None`");

            if !fields.is_empty() {
                write!(writer, "{{{}}}", fields)?;
            }
            write!(writer, ": ")?;

            Ok(())
        })?;

        // Write fields on the event
        ctx.field_format().format_fields(writer.by_ref(), event)?;

        writeln!(writer)
    }
}
