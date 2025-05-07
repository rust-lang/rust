//! A [tracing_subscriber::layer::Layer] that exports new-line delinated JSON.
//!
//! Usage:
//!
//! ```ignore
//! # use tracing_subscriber::Registry;
//! let layer = json::TimingLayer::new(std::io::stderr);
//! Registry::default().with(layer).init();
//! ```

use std::{io::Write as _, marker::PhantomData, time::Instant};

use ide_db::FxHashSet;
use tracing::{
    Event, Subscriber,
    span::{Attributes, Id},
};
use tracing_subscriber::{Layer, fmt::MakeWriter, layer::Context, registry::LookupSpan};

struct JsonData {
    name: &'static str,
    start: std::time::Instant,
}

impl JsonData {
    fn new(name: &'static str) -> Self {
        Self { name, start: Instant::now() }
    }
}

#[derive(Debug)]
pub(crate) struct TimingLayer<S, W> {
    writer: W,
    _inner: PhantomData<fn(S)>,
}

impl<S, W> TimingLayer<S, W> {
    pub(crate) fn new(writer: W) -> Self {
        Self { writer, _inner: PhantomData }
    }
}

impl<S, W> Layer<S> for TimingLayer<S, W>
where
    S: Subscriber + for<'span> LookupSpan<'span>,
    W: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();

        let data = JsonData::new(attrs.metadata().name());
        span.extensions_mut().insert(data);
    }

    fn on_event(&self, _event: &Event<'_>, _ctx: Context<'_, S>) {}

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        #[derive(serde_derive::Serialize)]
        struct JsonDataInner {
            name: &'static str,
            elapsed_ms: u128,
        }

        let span = ctx.span(&id).unwrap();
        let Some(data) = span.extensions_mut().remove::<JsonData>() else {
            return;
        };

        let data = JsonDataInner { name: data.name, elapsed_ms: data.start.elapsed().as_millis() };
        let mut out = serde_json::to_string(&data).expect("Unable to serialize data");
        out.push('\n');
        self.writer.make_writer().write_all(out.as_bytes()).expect("Unable to write data");
    }
}

#[derive(Default, Clone, Debug)]
pub(crate) struct JsonFilter {
    pub(crate) allowed_names: Option<FxHashSet<String>>,
}

impl JsonFilter {
    pub(crate) fn from_spec(spec: &str) -> Self {
        let allowed_names = if spec == "*" {
            None
        } else {
            Some(FxHashSet::from_iter(spec.split('|').map(String::from)))
        };

        Self { allowed_names }
    }
}
