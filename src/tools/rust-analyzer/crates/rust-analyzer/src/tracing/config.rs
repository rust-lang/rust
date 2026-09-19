//! Simple logger that logs either to stderr or to a file, using `tracing_subscriber`
//! filter syntax and `tracing_appender` for non blocking output.

use std::io;

use anyhow::Context;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{
    Layer, Registry,
    filter::{Targets, filter_fn},
    fmt::{MakeWriter, time},
    layer::SubscriberExt,
};
use tracing_tree::HierarchicalLayer;

use crate::tracing::hprof;
use crate::tracing::json;

#[derive(Debug)]
pub struct Config<T> {
    pub writer: T,
    pub filter: String,
    /// The meaning of SOLVER_DEBUG is to tell the solver crates
    /// (i.e. rustc_type_ir, rustc_next_trait_solver) how to filter tracing
    /// logs. But now we can only have just one filter, which means we have to
    /// merge the solver filter to our main filter (from RA_LOG env).
    ///
    /// The acceptable syntax of SOLVER_DEBUG is `target[span{field=value}]=level`.
    /// As the value should only affect chalk crates, we'd better manually
    /// specify the target. And for simplicity, SOLVER_DEBUG only accept the value
    /// that specify level.
    pub solver_filter: Option<String>,
    /// Filtering syntax, set in a shell:
    /// ```text
    /// env RA_PROFILE=*             // dump everything
    /// env RA_PROFILE=foo|bar|baz   // enabled only selected entries
    /// env RA_PROFILE=*@3>10        // dump everything, up to depth 3, if it takes more than 10
    /// ```
    pub profile_filter: Option<String>,

    /// Filtering syntax, set in a shell:
    /// ```text
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
            .with_writer(writer);

        let ra_fmt_layer = match time::OffsetTime::local_rfc_3339() {
            Ok(timer) => {
                // If we can get the time offset, format logs with the timezone.
                ra_fmt_layer.with_timer(timer).boxed()
            }
            Err(_) => {
                // Use system time if we can't get the time offset. This should
                // never happen on Linux, but can happen on e.g. OpenBSD.
                ra_fmt_layer.boxed()
            }
        }
        .with_filter(targets_filter);

        let solver_layer = match self.solver_filter {
            Some(solver_filter) => {
                let level: LevelFilter =
                    solver_filter.parse().with_context(|| "invalid solver log filter")?;

                // Once with `ra_ap_` and once without; in-tree r-a uses without, out-of-tree uses with.
                let solver_filter = Targets::new()
                    .with_target("ra_ap_rustc_type_ir", level)
                    .with_target("rustc_type_ir", level)
                    .with_target("ra_ap_rustc_next_trait_solver", level)
                    .with_target("rustc_next_trait_solver", level);
                // TODO: remove `.with_filter(LevelFilter::OFF)` on the `None` branch.
                HierarchicalLayer::default()
                    .with_indent_lines(true)
                    .with_ansi(false)
                    .with_indent_amount(2)
                    .with_writer(io::stderr)
                    .with_filter(solver_filter)
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
            .with(solver_layer);

        tracing::subscriber::set_global_default(subscriber)?;

        Ok(())
    }
}
