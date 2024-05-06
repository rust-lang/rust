//! Simple logger that logs either to stderr or to a file, using `tracing_subscriber`
//! filter syntax and `tracing_appender` for non blocking output.

use std::io;

use anyhow::Context;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{
    filter::Targets, fmt::MakeWriter, layer::SubscriberExt, util::SubscriberInitExt, Layer,
    Registry,
};
use tracing_tree::HierarchicalLayer;

use crate::tracing::hprof;

pub struct Config<T> {
    pub writer: T,
    pub filter: String,
    /// The meaning of CHALK_DEBUG is to tell chalk crates
    /// (i.e. chalk-solve, chalk-ir, chalk-recursive) how to filter tracing
    /// logs. But now we can only have just one filter, which means we have to
    /// merge chalk filter to our main filter (from RA_LOG env).
    ///
    /// The acceptable syntax of CHALK_DEBUG is `target[span{field=value}]=level`.
    /// As the value should only affect chalk crates, we'd better manually
    /// specify the target. And for simplicity, CHALK_DEBUG only accept the value
    /// that specify level.
    pub chalk_filter: Option<String>,
    /// Filtering syntax, set in a shell:
    /// ```
    /// env RA_PROFILE=*             // dump everything
    /// env RA_PROFILE=foo|bar|baz   // enabled only selected entries
    /// env RA_PROFILE=*@3>10        // dump everything, up to depth 3, if it takes more than 10
    /// ```
    pub profile_filter: Option<String>,
}

impl<T> Config<T>
where
    T: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
    pub fn init(self) -> anyhow::Result<()> {
        let filter: Targets = self
            .filter
            .parse()
            .with_context(|| format!("invalid log filter: `{}`", self.filter))?;

        let writer = self.writer;

        let ra_fmt_layer = tracing_subscriber::fmt::layer().with_writer(writer).with_filter(filter);

        let mut chalk_layer = None;
        if let Some(chalk_filter) = self.chalk_filter {
            let level: LevelFilter =
                chalk_filter.parse().with_context(|| "invalid chalk log filter")?;

            let chalk_filter = Targets::new()
                .with_target("chalk_solve", level)
                .with_target("chalk_ir", level)
                .with_target("chalk_recursive", level);
            chalk_layer = Some(
                HierarchicalLayer::default()
                    .with_indent_lines(true)
                    .with_ansi(false)
                    .with_indent_amount(2)
                    .with_writer(io::stderr)
                    .with_filter(chalk_filter),
            );
        };

        let profiler_layer = self.profile_filter.map(|spec| hprof::layer(&spec));

        Registry::default().with(ra_fmt_layer).with(chalk_layer).with(profiler_layer).try_init()?;

        Ok(())
    }
}
