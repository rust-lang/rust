//! Simple hierarchical profiler
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

use once_cell::sync::Lazy;

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
/// It returns a Profile structure and measure elapsed time between this method invocation and Profile structure drop.
/// It supports nested profiling scopes in case when this function invoked multiple times at the execution stack. In this case the profiling information will be nested at the output.
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
    let enabled = PROFILING_ENABLED.load(Ordering::Relaxed)
        && PROFILE_STACK.with(|stack| stack.borrow_mut().push(label));
    let label = if enabled { Some(label) } else { None };
    Profiler { label, detail: None }
}

pub struct Profiler {
    label: Option<Label>,
    detail: Option<String>,
}

impl Profiler {
    pub fn detail(mut self, detail: impl FnOnce() -> String) -> Profiler {
        if self.label.is_some() {
            self.detail = Some(detail())
        }
        self
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
    fn new(depth: usize, allowed: HashSet<String>, longer_than: Duration) -> Filter {
        Filter { depth, allowed, longer_than, version: 0 }
    }

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
        Filter::new(depth, allowed, longer_than)
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
    messages: Vec<Message>,
    filter: Filter,
}

struct Message {
    level: usize,
    duration: Duration,
    label: Label,
    detail: Option<String>,
}

impl ProfileStack {
    fn new() -> ProfileStack {
        ProfileStack { starts: Vec::new(), messages: Vec::new(), filter: Default::default() }
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
        true
    }

    pub fn pop(&mut self, label: Label, detail: Option<String>) {
        let start = self.starts.pop().unwrap();
        let duration = start.elapsed();
        let level = self.starts.len();
        self.messages.push(Message { level, duration, label, detail });
        if level == 0 {
            let stdout = stderr();
            let longer_than = self.filter.longer_than;
            // Convert to millis for comparison to avoid problems with rounding
            // (otherwise we could print `0ms` despite user's `>0` filter when
            // `duration` is just a few nanos).
            if duration.as_millis() > longer_than.as_millis() {
                print(&self.messages, longer_than, &mut stdout.lock());
            }
            self.messages.clear();
        }
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        match self {
            Profiler { label: Some(label), detail } => {
                PROFILE_STACK.with(|stack| {
                    stack.borrow_mut().pop(label, detail.take());
                });
            }
            Profiler { label: None, .. } => (),
        }
    }
}

fn print(msgs: &[Message], longer_than: Duration, out: &mut impl Write) {
    if msgs.is_empty() {
        return;
    }
    let children_map = idx_to_children(msgs);
    let root_idx = msgs.len() - 1;
    print_for_idx(root_idx, &children_map, msgs, longer_than, out);
}

fn print_for_idx(
    current_idx: usize,
    children_map: &[Vec<usize>],
    msgs: &[Message],
    longer_than: Duration,
    out: &mut impl Write,
) {
    let current = &msgs[current_idx];
    let current_indent = "    ".repeat(current.level);
    let detail = current.detail.as_ref().map(|it| format!(" @ {}", it)).unwrap_or_default();
    writeln!(
        out,
        "{}{:5}ms - {}{}",
        current_indent,
        current.duration.as_millis(),
        current.label,
        detail,
    )
    .expect("printing profiling info");

    let longer_than_millis = longer_than.as_millis();
    let children_indices = &children_map[current_idx];
    let mut accounted_for = Duration::default();
    let mut short_children = BTreeMap::new(); // Use `BTreeMap` to get deterministic output.

    for child_idx in children_indices.iter() {
        let child = &msgs[*child_idx];
        if child.duration.as_millis() > longer_than_millis {
            print_for_idx(*child_idx, children_map, msgs, longer_than, out);
        } else {
            let pair = short_children.entry(child.label).or_insert((Duration::default(), 0));
            pair.0 += child.duration;
            pair.1 += 1;
        }
        accounted_for += child.duration;
    }

    for (child_msg, (duration, count)) in short_children.iter() {
        let millis = duration.as_millis();
        writeln!(out, "    {}{:5}ms - {} ({} calls)", current_indent, millis, child_msg, count)
            .expect("printing profiling info");
    }

    let unaccounted_millis = (current.duration - accounted_for).as_millis();
    if !children_indices.is_empty()
        && unaccounted_millis > 0
        && unaccounted_millis > longer_than_millis
    {
        writeln!(out, "    {}{:5}ms - ???", current_indent, unaccounted_millis)
            .expect("printing profiling info");
    }
}

/// Returns a mapping from an index in the `msgs` to the vector with the indices of its children.
///
/// This assumes that the entries in `msgs` are in the order of when the calls to `profile` finish.
/// In other words, a postorder of the call graph. In particular, the root is the last element of
/// `msgs`.
fn idx_to_children(msgs: &[Message]) -> Vec<Vec<usize>> {
    // Initialize with the index of the root; `msgs` and `ancestors` should be never empty.
    assert!(!msgs.is_empty());
    let mut ancestors = vec![msgs.len() - 1];
    let mut result: Vec<Vec<usize>> = vec![vec![]; msgs.len()];
    for (idx, msg) in msgs[..msgs.len() - 1].iter().enumerate().rev() {
        // We need to find the parent of the current message, i.e., the last ancestor that has a
        // level lower than the current message.
        while msgs[*ancestors.last().unwrap()].level >= msg.level {
            ancestors.pop();
        }
        result[*ancestors.last().unwrap()].push(idx);
        ancestors.push(idx);
    }
    // Note that above we visited all children from the last to the first one. Let's reverse vectors
    // to get the more natural order where the first element is the first child.
    for vec in result.iter_mut() {
        vec.reverse();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_profile() {
        let s = vec!["profile1".to_string(), "profile2".to_string()];
        let f = Filter::new(2, s, Duration::new(0, 0));
        set_filter(f);
        profiling_function1();
    }

    fn profiling_function1() {
        let _p = profile("profile1");
        profiling_function2();
    }

    fn profiling_function2() {
        let _p = profile("profile2");
    }

    #[test]
    fn test_longer_than() {
        let mut result = vec![];
        let msgs = vec![
            Message { level: 1, duration: Duration::from_nanos(3), label: "bar", detail: None },
            Message { level: 1, duration: Duration::from_nanos(2), label: "bar", detail: None },
            Message { level: 0, duration: Duration::from_millis(1), label: "foo", detail: None },
        ];
        print(&msgs, Duration::from_millis(0), &mut result);
        // The calls to `bar` are so short that they'll be rounded to 0ms and should get collapsed
        // when printing.
        assert_eq!(
            std::str::from_utf8(&result).unwrap(),
            "    1ms - foo\n        0ms - bar (2 calls)\n"
        );
    }

    #[test]
    fn test_unaccounted_for_topmost() {
        let mut result = vec![];
        let msgs = vec![
            Message { level: 1, duration: Duration::from_millis(2), label: "bar", detail: None },
            Message { level: 0, duration: Duration::from_millis(5), label: "foo", detail: None },
        ];
        print(&msgs, Duration::from_millis(0), &mut result);
        assert_eq!(
            std::str::from_utf8(&result).unwrap().lines().collect::<Vec<_>>(),
            vec![
                "    5ms - foo",
                "        2ms - bar",
                "        3ms - ???",
                // Dummy comment to improve formatting
            ]
        );
    }

    #[test]
    fn test_unaccounted_for_multiple_levels() {
        let mut result = vec![];
        let msgs = vec![
            Message { level: 2, duration: Duration::from_millis(3), label: "baz", detail: None },
            Message { level: 1, duration: Duration::from_millis(5), label: "bar", detail: None },
            Message { level: 2, duration: Duration::from_millis(2), label: "baz", detail: None },
            Message { level: 1, duration: Duration::from_millis(4), label: "bar", detail: None },
            Message { level: 0, duration: Duration::from_millis(9), label: "foo", detail: None },
        ];
        print(&msgs, Duration::from_millis(0), &mut result);
        assert_eq!(
            std::str::from_utf8(&result).unwrap().lines().collect::<Vec<_>>(),
            vec![
                "    9ms - foo",
                "        5ms - bar",
                "            3ms - baz",
                "            2ms - ???",
                "        4ms - bar",
                "            2ms - baz",
                "            2ms - ???",
            ]
        );
    }
}
