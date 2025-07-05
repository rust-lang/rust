// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2020 Thoren Paulson
//! This file is taken unmodified from the following link, except for file attributes and
//! `extern crate` at the top.
//! https://github.com/thoren-d/tracing-chrome/blob/7e2625ab4aeeef2f0ef9bde9d6258dd181c04472/src/lib.rs
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

thread_local! {
    static OUT: RefCell<Option<Sender<Message>>> = const { RefCell::new(None) };
    static TID: RefCell<Option<usize>> = const { RefCell::new(None) };
}

type NameFn<S> = Box<dyn Fn(&EventOrSpan<'_, '_, S>) -> String + Send + Sync>;
type Object = serde_json::Map<String, JsonValue>;

/// A [`Layer`](tracing_subscriber::Layer) that writes a Chrome trace file.
pub struct ChromeLayer<S>
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    out: Arc<Mutex<Sender<Message>>>,
    start: std::time::Instant,
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
#[derive(Default)]
pub enum TraceStyle {
    /// Traces will be recorded as a group of threads.
    /// In this style, spans should be entered and exited on the same thread.
    #[default]
    Threaded,

    /// Traces will recorded as a group of asynchronous operations.
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
    Enter(f64, Callsite, Option<u64>),
    Event(f64, Callsite),
    Exit(f64, Callsite, Option<u64>),
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
        OUT.with(|val| val.replace(Some(tx.clone())));

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
            start: std::time::Instant::now(),
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

    fn get_tid(&self) -> (usize, bool) {
        TID.with(|value| {
            let tid = *value.borrow();
            match tid {
                Some(tid) => (tid, false),
                None => {
                    let tid = self.max_tid.fetch_add(1, Ordering::SeqCst);
                    value.replace(Some(tid));
                    (tid, true)
                }
            }
        })
    }

    fn get_callsite(&self, data: EventOrSpan<S>) -> Callsite {
        let (tid, new_thread) = self.get_tid();
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

        if new_thread {
            let name = match std::thread::current().name() {
                Some(name) => name.to_owned(),
                None => tid.to_string(),
            };
            self.send_message(Message::NewThread(tid, name));
        }

        Callsite {
            tid,
            name,
            target,
            file,
            line,
            args,
        }
    }

    fn get_root_id(span: SpanRef<S>) -> u64 {
        span.scope()
            .from_root()
            .take(1)
            .next()
            .unwrap_or(span)
            .id()
            .into_u64()
    }

    fn enter_span(&self, span: SpanRef<S>, ts: f64) {
        let callsite = self.get_callsite(EventOrSpan::Span(&span));
        let root_id = match self.trace_style {
            TraceStyle::Async => Some(ChromeLayer::get_root_id(span)),
            _ => None,
        };
        self.send_message(Message::Enter(ts, callsite, root_id));
    }

    fn exit_span(&self, span: SpanRef<S>, ts: f64) {
        let callsite = self.get_callsite(EventOrSpan::Span(&span));
        let root_id = match self.trace_style {
            TraceStyle::Async => Some(ChromeLayer::get_root_id(span)),
            _ => None,
        };
        self.send_message(Message::Exit(ts, callsite, root_id));
    }

    fn get_ts(&self) -> f64 {
        self.start.elapsed().as_nanos() as f64 / 1000.0
    }

    fn send_message(&self, message: Message) {
        OUT.with(move |val| {
            if val.borrow().is_some() {
                let _ignored = val.borrow().as_ref().unwrap().send(message);
            } else {
                let out = self.out.lock().unwrap().clone();
                let _ignored = out.send(message);
                val.replace(Some(out));
            }
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

        let ts = self.get_ts();
        self.enter_span(ctx.span(id).expect("Span not found."), ts);
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        if self.include_args {
            let span = ctx.span(id).unwrap();
            let mut exts = span.extensions_mut();

            let args = exts.get_mut::<ArgsWrapper>();

            if let Some(args) = args {
                let args = Arc::make_mut(&mut args.args);
                values.record(&mut JsonVisitor { object: args });
            }
        }
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let ts = self.get_ts();
        let callsite = self.get_callsite(EventOrSpan::Event(event));
        self.send_message(Message::Event(ts, callsite));
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let TraceStyle::Async = self.trace_style {
            return;
        }
        let ts = self.get_ts();
        self.exit_span(ctx.span(id).expect("Span not found."), ts);
    }

    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
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

        let ts = self.get_ts();
        self.enter_span(ctx.span(id).expect("Span not found."), ts);
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        if let TraceStyle::Threaded = self.trace_style {
            return;
        }

        let ts = self.get_ts();
        self.exit_span(ctx.span(&id).expect("Span not found."), ts);
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
