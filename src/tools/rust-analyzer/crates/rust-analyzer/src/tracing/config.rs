//! Simple logger that logs either to stderr or to a file, using `tracing_subscriber`
//! filter syntax and `tracing_appender` for non blocking output.

use std::io::{self};

use anyhow::Context;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{
    filter::{filter_fn, Targets},
    fmt::MakeWriter,
    layer::SubscriberExt,
    Layer, Registry,
};
use tracing_tree::HierarchicalLayer;

use crate::tracing::hprof;
use crate::tracing::json;

#[derive(Debug)]
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

    /// Filtering syntax, set in a shell:
    /// ```
    /// env RA_PROFILE_JSON=foo|bar|baz
    /// ```
    pub json_profile_filter: Option<String>,
}

impl<T> Config<T>
where
    T: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
    pub fn init(self) -> anyhow::Result<()> {
        let targets_filter: Targets = self
            .filter
            .parse()
            .with_context(|| format!("invalid log filter: `{}`", self.filter))?;

        let writer = self.writer;

        let ra_fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_ansi(false)
            .with_writer(writer)
            .with_filter(targets_filter);

        let chalk_layer = match self.chalk_filter {
            Some(chalk_filter) => {
                let level: LevelFilter =
                    chalk_filter.parse().with_context(|| "invalid chalk log filter")?;

                let chalk_filter = Targets::new()
                    .with_target("chalk_solve", level)
                    .with_target("chalk_ir", level)
                    .with_target("chalk_recursive", level);
                // TODO: remove `.with_filter(LevelFilter::OFF)` on the `None` branch.
                HierarchicalLayer::default()
                    .with_indent_lines(true)
                    .with_ansi(false)
                    .with_indent_amount(2)
                    .with_writer(io::stderr)
                    .with_filter(chalk_filter)
                    .boxed()
            }
            None => None::<HierarchicalLayer>.with_filter(LevelFilter::OFF).boxed(),
        };

        // TODO: remove `.with_filter(LevelFilter::OFF)` on the `None` branch.
        let profiler_layer = match self.profile_filter {
            Some(spec) => Some(hprof::SpanTree::new(&spec)).with_filter(LevelFilter::INFO),
            None => None.with_filter(LevelFilter::OFF),
        };

        let json_profiler_layer = match self.json_profile_filter {
            Some(spec) => {
                let filter = json::JsonFilter::from_spec(&spec);
                let filter = filter_fn(move |metadata| {
                    let allowed = match &filter.allowed_names {
                        Some(names) => names.contains(metadata.name()),
                        None => true,
                    };

                    allowed && metadata.is_span()
                });
                Some(json::TimingLayer::new(std::io::stderr).with_filter(filter))
            }
            None => None,
        };

        let subscriber = Registry::default()
            .with(ra_fmt_layer)
            .with(json_profiler_layer)
            .with(profiler_layer)
            .with(chalk_layer);

        tracing::subscriber::set_global_default(subscriber)?;

        Ok(())
    }
}
