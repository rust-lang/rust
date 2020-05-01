//! Simple hierarchical profiler
use once_cell::sync::Lazy;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    io::{stderr, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        RwLock,
    },
    time::{Duration, Instant},
};

use crate::tree::{Idx, Tree};

/// Filtering syntax
/// env RA_PROFILE=*             // dump everything
/// env RA_PROFILE=foo|bar|baz   // enabled only selected entries
/// env RA_PROFILE=*@3>10        // dump everything, up to depth 3, if it takes more than 10 ms
pub fn init() {
    let spec = std::env::var("RA_PROFILE").unwrap_or_default();
    init_from(&spec);
}

pub fn init_from(spec: &str) {
    let filter = if spec.is_empty() { Filter::disabled() } else { Filter::from_spec(spec) };
    filter.install();
}

pub type Label = &'static str;

/// This function starts a profiling scope in the current execution stack with a given description.
/// It returns a `Profile` struct that measures elapsed time between this method invocation and `Profile` struct drop.
/// It supports nested profiling scopes in case when this function is invoked multiple times at the execution stack.
/// In this case the profiling information will be nested at the output.
/// Profiling information is being printed in the stderr.
///
/// # Example
/// ```
/// use ra_prof::{profile, set_filter, Filter};
///
/// let f = Filter::from_spec("profile1|profile2@2");
/// set_filter(f);
/// profiling_function1();
///
/// fn profiling_function1() {
///     let _p = profile("profile1");
///     profiling_function2();
/// }
///
/// fn profiling_function2() {
///     let _p = profile("profile2");
/// }
/// ```
/// This will print in the stderr the following:
/// ```text
///  0ms - profile
///      0ms - profile2
/// ```
pub fn profile(label: Label) -> Profiler {
    assert!(!label.is_empty());

    if PROFILING_ENABLED.load(Ordering::Relaxed)
        && PROFILE_STACK.with(|stack| stack.borrow_mut().push(label))
    {
        Profiler(Some(ProfilerImpl { label, detail: None }))
    } else {
        Profiler(None)
    }
}

pub struct Profiler(Option<ProfilerImpl>);

struct ProfilerImpl {
    label: Label,
    detail: Option<String>,
}

impl Profiler {
    pub fn detail(mut self, detail: impl FnOnce() -> String) -> Profiler {
        if let Some(profiler) = &mut self.0 {
            profiler.detail = Some(detail())
        }
        self
    }
}

impl Drop for ProfilerImpl {
    fn drop(&mut self) {
        PROFILE_STACK.with(|it| it.borrow_mut().pop(self.label, self.detail.take()));
    }
}

static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
static FILTER: Lazy<RwLock<Filter>> = Lazy::new(Default::default);
thread_local!(static PROFILE_STACK: RefCell<ProfileStack> = RefCell::new(ProfileStack::new()));

#[derive(Default, Clone, Debug)]
struct Filter {
    depth: usize,
    allowed: HashSet<String>,
    longer_than: Duration,
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

        let depth = if let Some(idx) = spec.rfind('@') {
            let depth: usize = spec[idx + 1..].parse().expect("invalid profile depth");
            spec = &spec[..idx];
            depth
        } else {
            999
        };
        let allowed =
            if spec == "*" { HashSet::new() } else { spec.split('|').map(String::from).collect() };
        Filter { depth, allowed, longer_than, version: 0 }
    }

    fn install(mut self) {
        PROFILING_ENABLED.store(self.depth > 0, Ordering::SeqCst);
        let mut old = FILTER.write().unwrap();
        self.version = old.version + 1;
        *old = self;
    }
}

struct ProfileStack {
    starts: Vec<Instant>,
    filter: Filter,
    messages: Tree<Message>,
}

#[derive(Default)]
struct Message {
    duration: Duration,
    label: Label,
    detail: Option<String>,
}

impl ProfileStack {
    fn new() -> ProfileStack {
        ProfileStack { starts: Vec::new(), messages: Tree::default(), filter: Default::default() }
    }

    fn push(&mut self, label: Label) -> bool {
        if self.starts.is_empty() {
            if let Ok(f) = FILTER.try_read() {
                if f.version > self.filter.version {
                    self.filter = f.clone();
                }
            };
        }
        if self.starts.len() > self.filter.depth {
            return false;
        }
        let allowed = &self.filter.allowed;
        if self.starts.is_empty() && !allowed.is_empty() && !allowed.contains(label) {
            return false;
        }

        self.starts.push(Instant::now());
        self.messages.start();
        true
    }

    pub fn pop(&mut self, label: Label, detail: Option<String>) {
        let start = self.starts.pop().unwrap();
        let duration = start.elapsed();
        self.messages.finish(Message { duration, label, detail });
        if self.starts.is_empty() {
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
}

fn print(
    tree: &Tree<Message>,
    curr: Idx<Message>,
    level: u32,
    longer_than: Duration,
    out: &mut impl Write,
) {
    let current_indent = "    ".repeat(level as usize);
    let detail = tree[curr].detail.as_ref().map(|it| format!(" @ {}", it)).unwrap_or_default();
    writeln!(
        out,
        "{}{:5}ms - {}{}",
        current_indent,
        tree[curr].duration.as_millis(),
        tree[curr].label,
        detail,
    )
    .expect("printing profiling info");

    let mut accounted_for = Duration::default();
    let mut short_children = BTreeMap::new(); // Use `BTreeMap` to get deterministic output.
    for child in tree.children(curr) {
        accounted_for += tree[child].duration;

        if tree[child].duration.as_millis() > longer_than.as_millis() {
            print(tree, child, level + 1, longer_than, out)
        } else {
            let (total_duration, cnt) =
                short_children.entry(tree[child].label).or_insert((Duration::default(), 0));
            *total_duration += tree[child].duration;
            *cnt += 1;
        }
    }

    for (child_msg, (duration, count)) in short_children.iter() {
        let millis = duration.as_millis();
        writeln!(out, "    {}{:5}ms - {} ({} calls)", current_indent, millis, child_msg, count)
            .expect("printing profiling info");
    }

    let unaccounted = tree[curr].duration - accounted_for;
    if tree.children(curr).next().is_some() && unaccounted > longer_than {
        writeln!(out, "    {}{:5}ms - ???", current_indent, unaccounted.as_millis())
            .expect("printing profiling info");
    }
}
