//! FIXME: write short doc here

mod memory_usage;
#[cfg(feature = "cpu_profiler")]
mod google_cpu_profiler;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    collections::HashSet,
    io::{stderr, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        RwLock,
    },
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;

pub use crate::memory_usage::{Bytes, MemoryUsage};

// We use jemalloc mainly to get heap usage statistics, actual performance
// difference is not measures.
#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub fn init() {
    set_filter(match std::env::var("RA_PROFILE") {
        Ok(spec) => Filter::from_spec(&spec),
        Err(_) => Filter::disabled(),
    });
}

/// Set profiling filter. It specifies descriptions allowed to profile.
/// This is helpful when call stack has too many nested profiling scopes.
/// Additionally filter can specify maximum depth of profiling scopes nesting.
///
/// #Example
/// ```
/// use ra_prof::{set_filter, Filter};
/// let f = Filter::from_spec("profile1|profile2@2");
/// set_filter(f);
/// ```
pub fn set_filter(f: Filter) {
    PROFILING_ENABLED.store(f.depth > 0, Ordering::SeqCst);
    let set: HashSet<_> = f.allowed.iter().cloned().collect();
    let mut old = FILTER.write().unwrap();
    let filter_data = FilterData {
        depth: f.depth,
        allowed: set,
        longer_than: f.longer_than,
        version: old.version + 1,
    };
    *old = filter_data;
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
    if !PROFILING_ENABLED.load(Ordering::Relaxed) {
        return Profiler { label: None, detail: None };
    }

    PROFILE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.starts.is_empty() {
            if let Ok(f) = FILTER.try_read() {
                if f.version > stack.filter_data.version {
                    stack.filter_data = f.clone();
                }
            };
        }
        if stack.starts.len() > stack.filter_data.depth {
            return Profiler { label: None, detail: None };
        }
        let allowed = &stack.filter_data.allowed;
        if stack.starts.is_empty() && !allowed.is_empty() && !allowed.contains(label) {
            return Profiler { label: None, detail: None };
        }

        stack.starts.push(Instant::now());
        Profiler { label: Some(label), detail: None }
    })
}

pub fn print_time(label: Label) -> impl Drop {
    struct Guard {
        label: Label,
        start: Instant,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            eprintln!("{}: {:?}", self.label, self.start.elapsed())
        }
    }

    Guard { label, start: Instant::now() }
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

pub struct Filter {
    depth: usize,
    allowed: Vec<String>,
    longer_than: Duration,
}

impl Filter {
    // Filtering syntax
    // env RA_PROFILE=*             // dump everything
    // env RA_PROFILE=foo|bar|baz   // enabled only selected entries
    // env RA_PROFILE=*@3>10        // dump everything, up to depth 3, if it takes more than 10 ms
    pub fn from_spec(mut spec: &str) -> Filter {
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
            if spec == "*" { Vec::new() } else { spec.split('|').map(String::from).collect() };
        Filter::new(depth, allowed, longer_than)
    }

    pub fn disabled() -> Filter {
        Filter::new(0, Vec::new(), Duration::new(0, 0))
    }

    pub fn new(depth: usize, allowed: Vec<String>, longer_than: Duration) -> Filter {
        Filter { depth, allowed, longer_than }
    }
}

struct ProfileStack {
    starts: Vec<Instant>,
    messages: Vec<Message>,
    filter_data: FilterData,
}

struct Message {
    level: usize,
    duration: Duration,
    label: Label,
    detail: Option<String>,
}

impl ProfileStack {
    fn new() -> ProfileStack {
        ProfileStack { starts: Vec::new(), messages: Vec::new(), filter_data: Default::default() }
    }
}

#[derive(Default, Clone)]
struct FilterData {
    depth: usize,
    version: usize,
    allowed: HashSet<String>,
    longer_than: Duration,
}

static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);

static FILTER: Lazy<RwLock<FilterData>> = Lazy::new(Default::default);

thread_local!(static PROFILE_STACK: RefCell<ProfileStack> = RefCell::new(ProfileStack::new()));

impl Drop for Profiler {
    fn drop(&mut self) {
        match self {
            Profiler { label: Some(label), detail } => {
                PROFILE_STACK.with(|stack| {
                    let mut stack = stack.borrow_mut();
                    let start = stack.starts.pop().unwrap();
                    let duration = start.elapsed();
                    let level = stack.starts.len();
                    stack.messages.push(Message { level, duration, label, detail: detail.take() });
                    if level == 0 {
                        let stdout = stderr();
                        let longer_than = stack.filter_data.longer_than;
                        // Convert to millis for comparison to avoid problems with rounding
                        // (otherwise we could print `0ms` despite user's `>0` filter when
                        // `duration` is just a few nanos).
                        if duration.as_millis() > longer_than.as_millis() {
                            print(&stack.messages, longer_than, &mut stdout.lock());
                        }
                        stack.messages.clear();
                    }
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

/// Prints backtrace to stderr, useful for debugging.
#[cfg(feature = "backtrace")]
pub fn print_backtrace() {
    let bt = backtrace::Backtrace::new();
    eprintln!("{:?}", bt);
}

thread_local!(static IN_SCOPE: RefCell<bool> = RefCell::new(false));

/// Allows to check if the current code is withing some dynamic scope, can be
/// useful during debugging to figure out why a function is called.
pub struct Scope {
    prev: bool,
}

impl Scope {
    pub fn enter() -> Scope {
        let prev = IN_SCOPE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), true));
        Scope { prev }
    }
    pub fn is_active() -> bool {
        IN_SCOPE.with(|slot| *slot.borrow())
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        IN_SCOPE.with(|slot| *slot.borrow_mut() = self.prev);
    }
}

/// A wrapper around google_cpu_profiler.
///
/// Usage:
/// 1. Install gpref_tools (https://github.com/gperftools/gperftools), probably packaged with your Linux distro.
/// 2. Build with `cpu_profiler` feature.
/// 3. Tun the code, the *raw* output would be in the `./out.profile` file.
/// 4. Install pprof for visualization (https://github.com/google/pprof).
/// 5. Use something like `pprof -svg target/release/rust-analyzer ./out.profile` to see the results.
///
/// For example, here's how I run profiling on NixOS:
///
/// ```bash
/// $ nix-shell -p gperftools --run \
///     'cargo run --release -p rust-analyzer -- parse < ~/projects/rustbench/parser.rs > /dev/null'
/// ```
#[derive(Debug)]
pub struct CpuProfiler {
    _private: (),
}

pub fn cpu_profiler() -> CpuProfiler {
    #[cfg(feature = "cpu_profiler")]
    {
        google_cpu_profiler::start("./out.profile".as_ref())
    }

    #[cfg(not(feature = "cpu_profiler"))]
    {
        eprintln!("cpu_profiler feature is disabled")
    }

    CpuProfiler { _private: () }
}

impl Drop for CpuProfiler {
    fn drop(&mut self) {
        #[cfg(feature = "cpu_profiler")]
        {
            google_cpu_profiler::stop()
        }
    }
}

pub fn memory_usage() -> MemoryUsage {
    MemoryUsage::current()
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
