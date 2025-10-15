// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2020 Thoren Paulson
//! This file was initially taken from the following link:
//! <https://github.com/thoren-d/tracing-chrome/blob/7e2625ab4aeeef2f0ef9bde9d6258dd181c04472/src/lib.rs>
//!
//! The precise changes that were made to the original file can be found in git history
//! (`git log -- path/to/tracing_chrome.rs`), but in summary:
//! - the file attributes were changed and `extern crate` was added at the top
//! - if a tracing span has a field called "tracing_separate_thread", it will be given a separate
//!   span ID even in [TraceStyle::Threaded] mode, to make it appear on a separate line when viewing
//!   the trace in <https://ui.perfetto.dev>. This is the syntax to trigger this behavior:
//!   ```rust
//!   tracing::info_span!("my_span", tracing_separate_thread = tracing::field::Empty, /* ... */)
//!   ```
//! - use i64 instead of u64 for the "id" in [ChromeLayer::get_root_id] to be compatible with
//!   Perfetto
//! - use [ChromeLayer::with_elapsed_micros_subtracting_tracing] to make time measurements faster on
//!   Linux x86/x86_64 and to subtract time spent tracing from the timestamps in the trace file
//!
//! Depending on the tracing-chrome crate from crates.io is unfortunately not possible, since it
//! depends on `tracing_core` which conflicts with rustc_private's `tracing_core` (meaning it would
//! not be possible to use the [ChromeLayer] in a context that expects a [Layer] from
//! rustc_private's `tracing_core` version).
#![allow(warnings)]
#![cfg(feature = "tracing")]

// This is here and not in src/lib.rs since it is a direct dependency of tracing_chrome.rs and
// should not be included if the "tracing" feature is disabled.
extern crate tracing_core;

use tracing_core::{field::Field, span, Event, Subscriber};
use tracing_subscriber::{
    layer::Context,
    registry::{LookupSpan, SpanRef},
    Layer,
};

use serde_json::{json, Value as JsonValue};
use std::{
    marker::PhantomData,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use std::io::{BufWriter, Write};
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::{
    cell::{Cell, RefCell},
    thread::JoinHandle,
};

use crate::log::tracing_chrome_instant::TracingChromeInstant;

/// Contains thread-local data for threads that send tracing spans or events.
struct ThreadData {
    /// A unique ID for this thread, will populate "tid" field in the output trace file.
    tid: usize,
    /// A clone of [ChromeLayer::out] to avoid the expensive operation of accessing a mutex
    /// every time. This is used to send [Message]s to the thread that saves trace data to file.
    out: Sender<Message>,
    /// The instant in time this thread was started. All events happening on this thread will be
    /// saved to the trace file with a timestamp (the "ts" field) measured relative to this instant.
    start: TracingChromeInstant,
}

thread_local! {
    static THREAD_DATA: RefCell<Option<ThreadData>> = const { RefCell::new(None) };
}

type NameFn<S> = Box<dyn Fn(&EventOrSpan<'_, '_, S>) -> String + Send + Sync>;
type Object = serde_json::Map<String, JsonValue>;

/// A [`Layer`](tracing_subscriber::Layer) that writes a Chrome trace file.
pub struct ChromeLayer<S>
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    out: Arc<Mutex<Sender<Message>>>,
    max_tid: AtomicUsize,
    include_args: bool,
    include_locations: bool,
    trace_style: TraceStyle,
    name_fn: Option<NameFn<S>>,
    cat_fn: Option<NameFn<S>>,
    _inner: PhantomData<S>,
}

/// A builder for [`ChromeLayer`](crate::ChromeLayer).
#[derive(Default)]
pub struct ChromeLayerBuilder<S>
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    out_writer: Option<Box<dyn Write + Send>>,
    name_fn: Option<NameFn<S>>,
    cat_fn: Option<NameFn<S>>,
    include_args: bool,
    include_locations: bool,
    trace_style: TraceStyle,
    _inner: PhantomData<S>,
}

/// Decides how traces will be recorded.
/// Also see <https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.jh64i9l3vwa1>
#[derive(Default)]
pub enum TraceStyle {
    /// Traces will be recorded as a group of threads, and all spans on the same thread will appear
    /// on a single trace line in <https://ui.perfetto.dev>.
    /// In this style, spans should be entered and exited on the same thread.
    ///
    /// If a tracing span has a field called "tracing_separate_thread", it will be given a separate
    /// span ID even in this mode, to make it appear on a separate line when viewing the trace in
    /// <https://ui.perfetto.dev>. This is the syntax to trigger this behavior:
    /// ```rust
    /// tracing::info_span!("my_span", tracing_separate_thread = tracing::field::Empty, /* ... */)
    /// ```
    /// [tracing::field::Empty] is used so that other tracing layers (e.g. the logger) will ignore
    /// the "tracing_separate_thread" argument and not print out anything for it.
    #[default]
    Threaded,

    /// Traces will recorded as a group of asynchronous operations. All spans will be given separate
    /// span IDs and will appear on separate trace lines in <https://ui.perfetto.dev>.
    Async,
}

impl<S> ChromeLayerBuilder<S>
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    pub fn new() -> Self {
        ChromeLayerBuilder {
            out_writer: None,
            name_fn: None,
            cat_fn: None,
            include_args: false,
            include_locations: true,
            trace_style: TraceStyle::Threaded,
            _inner: PhantomData,
        }
    }

    /// Set the file to which to output the trace.
    ///
    /// Defaults to `./trace-{unix epoch in micros}.json`.
    ///
    /// # Panics
    ///
    /// If `file` could not be opened/created. To handle errors,
    /// open a file and pass it to [`writer`](crate::ChromeLayerBuilder::writer) instead.
    pub fn file<P: AsRef<Path>>(self, file: P) -> Self {
        self.writer(std::fs::File::create(file).expect("Failed to create trace file."))
    }

    /// Supply an arbitrary writer to which to write trace contents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tracing_chrome::ChromeLayerBuilder;
    /// # use tracing_subscriber::prelude::*;
    /// let (layer, guard) = ChromeLayerBuilder::new().writer(std::io::sink()).build();
    /// # tracing_subscriber::registry().with(layer).init();
    /// ```
    pub fn writer<W: Write + Send + 'static>(mut self, writer: W) -> Self {
        self.out_writer = Some(Box::new(writer));
        self
    }

    /// Include arguments in each trace entry.
    ///
    /// Defaults to `false`.
    ///
    /// Includes the arguments used when creating a span/event
    /// in the "args" section of the trace entry.
    pub fn include_args(mut self, include: bool) -> Self {
        self.include_args = include;
        self
    }

    /// Include file+line with each trace entry.
    ///
    /// Defaults to `true`.
    ///
    /// This can add quite a bit of data to the output so turning
    /// it off might be helpful when collecting larger traces.
    pub fn include_locations(mut self, include: bool) -> Self {
        self.include_locations = include;
        self
    }

    /// Sets the style used when recording trace events.
    ///
    /// See [`TraceStyle`](crate::TraceStyle) for details.
    pub fn trace_style(mut self, style: TraceStyle) -> Self {
        self.trace_style = style;
        self
    }

    /// Allows supplying a function that derives a name from
    /// an Event or Span. The result is used as the "name" field
    /// on trace entries.
    ///
    /// # Example
    /// ```
    /// use tracing_chrome::{ChromeLayerBuilder, EventOrSpan};
    /// use tracing_subscriber::{registry::Registry, prelude::*};
    ///
    /// let (chrome_layer, _guard) = ChromeLayerBuilder::new().name_fn(Box::new(|event_or_span| {
    ///     match event_or_span {
    ///         EventOrSpan::Event(ev) => { ev.metadata().name().into() },
    ///         EventOrSpan::Span(_s) => { "span".into() },
    ///     }
    /// })).build();
    /// tracing_subscriber::registry().with(chrome_layer).init()
    /// ```
    pub fn name_fn(mut self, name_fn: NameFn<S>) -> Self {
        self.name_fn = Some(name_fn);
        self
    }

    /// Allows supplying a function that derives a category from
    /// an Event or Span. The result is used as the "cat" field on
    /// trace entries.
    ///
    /// # Example
    /// ```
    /// use tracing_chrome::{ChromeLayerBuilder, EventOrSpan};
    /// use tracing_subscriber::{registry::Registry, prelude::*};
    ///
    /// let (chrome_layer, _guard) = ChromeLayerBuilder::new().category_fn(Box::new(|_| {
    ///     "my_module".into()
    /// })).build();
    /// tracing_subscriber::registry().with(chrome_layer).init()
    /// ```
    pub fn category_fn(mut self, cat_fn: NameFn<S>) -> Self {
        self.cat_fn = Some(cat_fn);
        self
    }

    /// Creates a [`ChromeLayer`](crate::ChromeLayer) and associated [`FlushGuard`](crate::FlushGuard).
    ///
    /// # Panics
    ///
    /// If no file or writer was specified and the default trace file could not be opened/created.
    pub fn build(self) -> (ChromeLayer<S>, FlushGuard) {
        ChromeLayer::new(self)
    }
}

/// This guard will signal the thread writing the trace file to stop and join it when dropped.
pub struct FlushGuard {
    sender: Sender<Message>,
    handle: Cell<Option<JoinHandle<()>>>,
}

impl FlushGuard {
    /// Signals the trace writing thread to flush to disk.
    pub fn flush(&self) {
        if let Some(handle) = self.handle.take() {
            let _ignored = self.sender.send(Message::Flush);
            self.handle.set(Some(handle));
        }
    }

    /// Finishes the current trace and starts a new one.
    ///
    /// If a [`Write`](std::io::Write) implementation is supplied,
    /// the new trace is written to it. Otherwise, the new trace
    /// goes to `./trace-{unix epoc in micros}.json`.
    pub fn start_new(&self, writer: Option<Box<dyn Write + Send>>) {
        if let Some(handle) = self.handle.take() {
            let _ignored = self.sender.send(Message::StartNew(writer));
            self.handle.set(Some(handle));
        }
    }
}

impl Drop for FlushGuard {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ignored = self.sender.send(Message::Drop);
            if handle.join().is_err() {
                eprintln!("tracing_chrome: Trace writing thread panicked.");
            }
        }
    }
}

struct Callsite {
    tid: usize,
    name: String,
    target: String,
    file: Option<&'static str>,
    line: Option<u32>,
    args: Option<Arc<Object>>,
}

enum Message {
    Enter(f64, Callsite, Option<i64>),
    Event(f64, Callsite),
    Exit(f64, Callsite, Option<i64>),
    NewThread(usize, String),
    Flush,
    Drop,
    StartNew(Option<Box<dyn Write + Send>>),
}

/// Represents either an [`Event`](tracing_core::Event) or [`SpanRef`](tracing_subscriber::registry::SpanRef).
pub enum EventOrSpan<'a, 'b, S>
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    Event(&'a Event<'b>),
    Span(&'a SpanRef<'b, S>),
}

fn create_default_writer() -> Box<dyn Write + Send> {
    Box::new(
        std::fs::File::create(format!(
            "./trace-{}.json",
            std::time::SystemTime::UNIX_EPOCH
                .elapsed()
                .unwrap()
                .as_micros()
        ))
        .expect("Failed to create trace file."),
    )
}

impl<S> ChromeLayer<S>
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    fn new(mut builder: ChromeLayerBuilder<S>) -> (ChromeLayer<S>, FlushGuard) {
        let (tx, rx) = mpsc::channel();

        let out_writer = builder
            .out_writer
            .unwrap_or_else(|| create_default_writer());

        let handle = std::thread::spawn(move || {
            let mut write = BufWriter::new(out_writer);
            write.write_all(b"[\n").unwrap();

            let mut has_started = false;
            let mut thread_names: Vec<(usize, String)> = Vec::new();
            for msg in rx {
                if let Message::Flush = &msg {
                    write.flush().unwrap();
                    continue;
                } else if let Message::Drop = &msg {
                    break;
                } else if let Message::StartNew(writer) = msg {
                    // Finish off current file
                    write.write_all(b"\n]").unwrap();
                    write.flush().unwrap();

                    // Get or create new writer
                    let out_writer = writer.unwrap_or_else(|| create_default_writer());
                    write = BufWriter::new(out_writer);
                    write.write_all(b"[\n").unwrap();
                    has_started = false;

                    // Write saved thread names
                    for (tid, name) in thread_names.iter() {
                        let entry = json!({
                            "ph": "M",
                            "pid": 1,
                            "name": "thread_name",
                            "tid": *tid,
                            "args": {
                                "name": name,
                            },
                        });

                        if has_started {
                            write.write_all(b",\n").unwrap();
                        }
                        serde_json::to_writer(&mut write, &entry).unwrap();
                        has_started = true;
                    }
                    continue;
                }

                let (ph, ts, callsite, id) = match &msg {
                    Message::Enter(ts, callsite, None) => ("B", Some(ts), Some(callsite), None),
                    Message::Enter(ts, callsite, Some(root_id)) => {
                        ("b", Some(ts), Some(callsite), Some(root_id))
                    }
                    Message::Event(ts, callsite) => ("i", Some(ts), Some(callsite), None),
                    Message::Exit(ts, callsite, None) => ("E", Some(ts), Some(callsite), None),
                    Message::Exit(ts, callsite, Some(root_id)) => {
                        ("e", Some(ts), Some(callsite), Some(root_id))
                    }
                    Message::NewThread(_tid, _name) => ("M", None, None, None),
                    Message::Flush | Message::Drop | Message::StartNew(_) => {
                        panic!("Was supposed to break by now.")
                    }
                };
                let mut entry = json!({
                    "ph": ph,
                    "pid": 1,
                });

                if let Message::NewThread(tid, name) = msg {
                    thread_names.push((tid, name.clone()));
                    entry["name"] = "thread_name".into();
                    entry["tid"] = tid.into();
                    entry["args"] = json!({ "name": name });
                } else {
                    let ts = ts.unwrap();
                    let callsite = callsite.unwrap();
                    entry["ts"] = (*ts).into();
                    entry["name"] = callsite.name.clone().into();
                    entry["cat"] = callsite.target.clone().into();
                    entry["tid"] = callsite.tid.into();

                    if let Some(&id) = id {
                        entry["id"] = id.into();
                    }

                    if ph == "i" {
                        entry["s"] = "t".into();
                    }

                    if let (Some(file), Some(line)) = (callsite.file, callsite.line) {
                        entry[".file"] = file.into();
                        entry[".line"] = line.into();
                    }

                    if let Some(call_args) = &callsite.args {
                        if !call_args.is_empty() {
                            entry["args"] = (**call_args).clone().into();
                        }
                    }
                }

                if has_started {
                    write.write_all(b",\n").unwrap();
                }
                serde_json::to_writer(&mut write, &entry).unwrap();
                has_started = true;
            }

            write.write_all(b"\n]").unwrap();
            write.flush().unwrap();
        });

        let guard = FlushGuard {
            sender: tx.clone(),
            handle: Cell::new(Some(handle)),
        };
        let layer = ChromeLayer {
            out: Arc::new(Mutex::new(tx)),
            max_tid: AtomicUsize::new(0),
            name_fn: builder.name_fn.take(),
            cat_fn: builder.cat_fn.take(),
            include_args: builder.include_args,
            include_locations: builder.include_locations,
            trace_style: builder.trace_style,
            _inner: PhantomData,
        };

        (layer, guard)
    }

    fn get_callsite(&self, data: EventOrSpan<S>, tid: usize) -> Callsite {
        let name = self.name_fn.as_ref().map(|name_fn| name_fn(&data));
        let target = self.cat_fn.as_ref().map(|cat_fn| cat_fn(&data));
        let meta = match data {
            EventOrSpan::Event(e) => e.metadata(),
            EventOrSpan::Span(s) => s.metadata(),
        };
        let args = match data {
            EventOrSpan::Event(e) => {
                if self.include_args {
                    let mut args = Object::new();
                    e.record(&mut JsonVisitor { object: &mut args });
                    Some(Arc::new(args))
                } else {
                    None
                }
            }
            EventOrSpan::Span(s) => s
                .extensions()
                .get::<ArgsWrapper>()
                .map(|e| &e.args)
                .cloned(),
        };
        let name = name.unwrap_or_else(|| meta.name().into());
        let target = target.unwrap_or_else(|| meta.target().into());
        let (file, line) = if self.include_locations {
            (meta.file(), meta.line())
        } else {
            (None, None)
        };

        Callsite {
            tid,
            name,
            target,
            file,
            line,
            args,
        }
    }

    fn get_root_id(&self, span: SpanRef<S>) -> Option<i64> {
        // Returns `Option<i64>` instead of `Option<u64>` because apparently Perfetto gives an
        // error if an id does not fit in a 64-bit signed integer in 2's complement. We cast the
        // span id from `u64` to `i64` with wraparound, since negative values are fine.
        match self.trace_style {
            TraceStyle::Threaded => {
                if span.fields().field("tracing_separate_thread").is_some() {
                    // assign an independent "id" to spans with argument "tracing_separate_thread",
                    // so they appear a separate trace line in trace visualization tools, see
                    // https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.jh64i9l3vwa1
                    Some(span.id().into_u64().cast_signed()) // the comment above explains the cast
                } else {
                    None
                }
            },
            TraceStyle::Async => Some(
                span.scope()
                    .from_root()
                    .take(1)
                    .next()
                    .unwrap_or(span)
                    .id()
                    .into_u64()
                    .cast_signed() // the comment above explains the cast
            ),
        }
    }

    fn enter_span(&self, span: SpanRef<S>, ts: f64, tid: usize, out: &Sender<Message>) {
        let callsite = self.get_callsite(EventOrSpan::Span(&span), tid);
        let root_id = self.get_root_id(span);
        let _ignored = out.send(Message::Enter(ts, callsite, root_id));
    }

    fn exit_span(&self, span: SpanRef<S>, ts: f64, tid: usize, out: &Sender<Message>) {
        let callsite = self.get_callsite(EventOrSpan::Span(&span), tid);
        let root_id = self.get_root_id(span);
        let _ignored = out.send(Message::Exit(ts, callsite, root_id));
    }

    /// Helper function that measures how much time is spent while executing `f` and accounts for it
    /// in subsequent calls, with the aim to reduce biases in the data collected by `tracing_chrome`
    /// by subtracting the time spent inside tracing functions from the timeline. This makes it so
    /// that the time spent inside the `tracing_chrome` functions does not impact the timestamps
    /// inside the trace file (i.e. `ts`), even if such functions are slow (e.g. because they need
    /// to format arguments on the same thread those arguments are collected on, otherwise memory
    /// safety would be broken).
    ///
    /// `f` is called with the microseconds elapsed since the current thread was started (**not**
    /// since the program start!), with the current thread ID (i.e. `tid`), and with a [Sender] that
    /// can be used to send a [Message] to the thread that collects [Message]s and saves them to the
    /// trace file.
    #[inline(always)]
    fn with_elapsed_micros_subtracting_tracing(&self, f: impl Fn(f64, usize, &Sender<Message>)) {
        THREAD_DATA.with(|value| {
            let mut thread_data = value.borrow_mut();
            let (ThreadData { tid, out, start }, new_thread) = match thread_data.as_mut() {
                Some(thread_data) => (thread_data, false),
                None => {
                    let tid = self.max_tid.fetch_add(1, Ordering::SeqCst);
                    let out = self.out.lock().unwrap().clone();
                    let start = TracingChromeInstant::setup_for_thread_and_start(tid);
                    *thread_data = Some(ThreadData { tid, out, start });
                    (thread_data.as_mut().unwrap(), true)
                }
            };

            start.with_elapsed_micros_subtracting_tracing(|ts| {
                if new_thread {
                    let name = match std::thread::current().name() {
                        Some(name) => name.to_owned(),
                        None => tid.to_string(),
                    };
                    let _ignored = out.send(Message::NewThread(*tid, name));
                }
                f(ts, *tid, out);
            });
        });
    }
}

impl<S> Layer<S> for ChromeLayer<S>
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let TraceStyle::Async = self.trace_style {
            return;
        }

        self.with_elapsed_micros_subtracting_tracing(|ts, tid, out| {
            self.enter_span(ctx.span(id).expect("Span not found."), ts, tid, out);
        });
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        if self.include_args {
            self.with_elapsed_micros_subtracting_tracing(|_, _, _| {
                let span = ctx.span(id).unwrap();
                let mut exts = span.extensions_mut();

                let args = exts.get_mut::<ArgsWrapper>();

                if let Some(args) = args {
                    let args = Arc::make_mut(&mut args.args);
                    values.record(&mut JsonVisitor { object: args });
                }
            });
        }
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        self.with_elapsed_micros_subtracting_tracing(|ts, tid, out| {
            let callsite = self.get_callsite(EventOrSpan::Event(event), tid);
            let _ignored = out.send(Message::Event(ts, callsite));
        });
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let TraceStyle::Async = self.trace_style {
            return;
        }
        self.with_elapsed_micros_subtracting_tracing(|ts, tid, out| {
            self.exit_span(ctx.span(id).expect("Span not found."), ts, tid, out);
        });
    }

    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        self.with_elapsed_micros_subtracting_tracing(|ts, tid, out| {
            if self.include_args {
                let mut args = Object::new();
                attrs.record(&mut JsonVisitor { object: &mut args });
                ctx.span(id).unwrap().extensions_mut().insert(ArgsWrapper {
                    args: Arc::new(args),
                });
            }
            if let TraceStyle::Threaded = self.trace_style {
                return;
            }

            self.enter_span(ctx.span(id).expect("Span not found."), ts, tid, out);
        });
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        if let TraceStyle::Threaded = self.trace_style {
            return;
        }

        self.with_elapsed_micros_subtracting_tracing(|ts, tid, out| {
            self.exit_span(ctx.span(&id).expect("Span not found."), ts, tid, out);
        });
    }
}

struct JsonVisitor<'a> {
    object: &'a mut Object,
}

impl<'a> tracing_subscriber::field::Visit for JsonVisitor<'a> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.object
            .insert(field.name().to_owned(), format!("{value:?}").into());
    }
}

struct ArgsWrapper {
    args: Arc<Object>,
}
