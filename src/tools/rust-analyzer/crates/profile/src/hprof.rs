//! Simple hierarchical profiler
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    env, fmt,
    io::{stderr, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        RwLock,
    },
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;

use crate::tree::{Idx, Tree};

/// Filtering syntax
/// env RA_PROFILE=*             // dump everything
/// env RA_PROFILE=foo|bar|baz   // enabled only selected entries
/// env RA_PROFILE=*@3>10        // dump everything, up to depth 3, if it takes more than 10 ms
pub fn init() {
    countme::enable(env::var("RA_COUNT").is_ok());
    let spec = env::var("RA_PROFILE").unwrap_or_default();
    init_from(&spec);
}

pub fn init_from(spec: &str) {
    let filter = if spec.is_empty() { Filter::disabled() } else { Filter::from_spec(spec) };
    filter.install();
}

type Label = &'static str;

/// This function starts a profiling scope in the current execution stack with a given description.
/// It returns a `Profile` struct that measures elapsed time between this method invocation and `Profile` struct drop.
/// It supports nested profiling scopes in case when this function is invoked multiple times at the execution stack.
/// In this case the profiling information will be nested at the output.
/// Profiling information is being printed in the stderr.
///
/// # Example
/// ```
/// profile::init_from("profile1|profile2@2");
/// profiling_function1();
///
/// fn profiling_function1() {
///     let _p = profile::span("profile1");
///     profiling_function2();
/// }
///
/// fn profiling_function2() {
///     let _p = profile::span("profile2");
/// }
/// ```
/// This will print in the stderr the following:
/// ```text
///  0ms - profile
///      0ms - profile2
/// ```
#[inline]
pub fn span(label: Label) -> ProfileSpan {
    debug_assert!(!label.is_empty());

    let enabled = PROFILING_ENABLED.load(Ordering::Relaxed);
    if enabled && with_profile_stack(|stack| stack.push(label)) {
        ProfileSpan(Some(ProfilerImpl { label, detail: None }))
    } else {
        ProfileSpan(None)
    }
}

#[inline]
pub fn heartbeat_span() -> HeartbeatSpan {
    let enabled = PROFILING_ENABLED.load(Ordering::Relaxed);
    HeartbeatSpan::new(enabled)
}

#[inline]
pub fn heartbeat() {
    let enabled = PROFILING_ENABLED.load(Ordering::Relaxed);
    if enabled {
        with_profile_stack(|it| it.heartbeat(1));
    }
}

pub struct ProfileSpan(Option<ProfilerImpl>);

struct ProfilerImpl {
    label: Label,
    detail: Option<String>,
}

impl ProfileSpan {
    pub fn detail(mut self, detail: impl FnOnce() -> String) -> ProfileSpan {
        if let Some(profiler) = &mut self.0 {
            profiler.detail = Some(detail());
        }
        self
    }
}

impl Drop for ProfilerImpl {
    #[inline]
    fn drop(&mut self) {
        with_profile_stack(|it| it.pop(self.label, self.detail.take()));
    }
}

pub struct HeartbeatSpan {
    enabled: bool,
}

impl HeartbeatSpan {
    #[inline]
    pub fn new(enabled: bool) -> Self {
        if enabled {
            with_profile_stack(|it| it.heartbeats(true));
        }
        Self { enabled }
    }
}

impl Drop for HeartbeatSpan {
    fn drop(&mut self) {
        if self.enabled {
            with_profile_stack(|it| it.heartbeats(false));
        }
    }
}

static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
static FILTER: Lazy<RwLock<Filter>> = Lazy::new(Default::default);

fn with_profile_stack<T>(f: impl FnOnce(&mut ProfileStack) -> T) -> T {
    thread_local!(static STACK: RefCell<ProfileStack> = RefCell::new(ProfileStack::new()));
    STACK.with(|it| f(&mut it.borrow_mut()))
}

#[derive(Default, Clone, Debug)]
struct Filter {
    depth: usize,
    allowed: HashSet<String>,
    longer_than: Duration,
    heartbeat_longer_than: Duration,
    version: usize,
}

impl Filter {
    fn disabled() -> Filter {
        Filter::default()
    }

    fn from_spec(mut spec: &str) -> Filter {
        let longer_than = if let Some(idx) = spec.rfind('>') {
            let longer_than = spec[idx + 1..].parse().expect("invalid profile longer_than");
            spec = &spec[..idx];
            Duration::from_millis(longer_than)
        } else {
            Duration::new(0, 0)
        };
        let heartbeat_longer_than = longer_than;

        let depth = if let Some(idx) = spec.rfind('@') {
            let depth: usize = spec[idx + 1..].parse().expect("invalid profile depth");
            spec = &spec[..idx];
            depth
        } else {
            999
        };
        let allowed =
            if spec == "*" { HashSet::new() } else { spec.split('|').map(String::from).collect() };
        Filter { depth, allowed, longer_than, heartbeat_longer_than, version: 0 }
    }

    fn install(mut self) {
        PROFILING_ENABLED.store(self.depth > 0, Ordering::SeqCst);
        let mut old = FILTER.write().unwrap();
        self.version = old.version + 1;
        *old = self;
    }
}

struct ProfileStack {
    frames: Vec<Frame>,
    filter: Filter,
    messages: Tree<Message>,
    heartbeats: bool,
}

struct Frame {
    t: Instant,
    heartbeats: u32,
}

#[derive(Default)]
struct Message {
    duration: Duration,
    label: Label,
    detail: Option<String>,
}

impl ProfileStack {
    fn new() -> ProfileStack {
        ProfileStack {
            frames: Vec::new(),
            messages: Tree::default(),
            filter: Default::default(),
            heartbeats: false,
        }
    }

    fn push(&mut self, label: Label) -> bool {
        if self.frames.is_empty() {
            if let Ok(f) = FILTER.try_read() {
                if f.version > self.filter.version {
                    self.filter = f.clone();
                }
            };
        }
        if self.frames.len() > self.filter.depth {
            return false;
        }
        let allowed = &self.filter.allowed;
        if self.frames.is_empty() && !allowed.is_empty() && !allowed.contains(label) {
            return false;
        }

        self.frames.push(Frame { t: Instant::now(), heartbeats: 0 });
        self.messages.start();
        true
    }

    fn pop(&mut self, label: Label, detail: Option<String>) {
        let frame = self.frames.pop().unwrap();
        let duration = frame.t.elapsed();

        if self.heartbeats {
            self.heartbeat(frame.heartbeats);
            let avg_span = duration / (frame.heartbeats + 1);
            if avg_span > self.filter.heartbeat_longer_than {
                eprintln!("Too few heartbeats {label} ({}/{duration:?})?", frame.heartbeats);
            }
        }

        self.messages.finish(Message { duration, label, detail });
        if self.frames.is_empty() {
            let longer_than = self.filter.longer_than;
            // Convert to millis for comparison to avoid problems with rounding
            // (otherwise we could print `0ms` despite user's `>0` filter when
            // `duration` is just a few nanos).
            if duration.as_millis() > longer_than.as_millis() {
                if let Some(root) = self.messages.root() {
                    print(&self.messages, root, 0, longer_than, &mut stderr().lock());
                }
            }
            self.messages.clear();
        }
    }

    fn heartbeats(&mut self, yes: bool) {
        self.heartbeats = yes;
    }
    fn heartbeat(&mut self, n: u32) {
        if let Some(frame) = self.frames.last_mut() {
            frame.heartbeats += n;
        }
    }
}

fn print(
    tree: &Tree<Message>,
    curr: Idx<Message>,
    level: u32,
    longer_than: Duration,
    out: &mut impl Write,
) {
    let current_indent = "    ".repeat(level as usize);
    let detail = tree[curr].detail.as_ref().map(|it| format!(" @ {it}")).unwrap_or_default();
    writeln!(
        out,
        "{}{} - {}{}",
        current_indent,
        ms(tree[curr].duration),
        tree[curr].label,
        detail,
    )
    .expect("printing profiling info");

    let mut accounted_for = Duration::default();
    let mut short_children = BTreeMap::new(); // Use `BTreeMap` to get deterministic output.
    for child in tree.children(curr) {
        accounted_for += tree[child].duration;

        if tree[child].duration.as_millis() > longer_than.as_millis() {
            print(tree, child, level + 1, longer_than, out);
        } else {
            let (total_duration, cnt) =
                short_children.entry(tree[child].label).or_insert((Duration::default(), 0));
            *total_duration += tree[child].duration;
            *cnt += 1;
        }
    }

    for (child_msg, (duration, count)) in &short_children {
        writeln!(out, "    {current_indent}{} - {child_msg} ({count} calls)", ms(*duration))
            .expect("printing profiling info");
    }

    let unaccounted = tree[curr].duration - accounted_for;
    if tree.children(curr).next().is_some() && unaccounted > longer_than {
        writeln!(out, "    {current_indent}{} - ???", ms(unaccounted))
            .expect("printing profiling info");
    }
}

#[allow(non_camel_case_types)]
struct ms(Duration);

impl fmt::Display for ms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_millis() {
            0 => f.write_str("    0  "),
            n => write!(f, "{n:5}ms"),
        }
    }
}
