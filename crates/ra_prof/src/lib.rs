mod memory_usage;
#[cfg(feature = "cpu_profiler")]
mod google_cpu_profiler;

use std::{
    cell::RefCell,
    collections::HashSet,
    io::{stderr, Write},
    iter::repeat,
    mem,
    sync::{
        atomic::{AtomicBool, Ordering},
        RwLock,
    },
    time::{Duration, Instant},
};

use itertools::Itertools;
use once_cell::sync::Lazy;

pub use crate::memory_usage::{Bytes, MemoryUsage};

// We use jemalloc mainly to get heap usage statistics, actual performance
// difference is not measures.
#[cfg(feature = "jemalloc")]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

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
pub fn profile(desc: &str) -> Profiler {
    assert!(!desc.is_empty());
    if !PROFILING_ENABLED.load(Ordering::Relaxed) {
        return Profiler { desc: None };
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
            return Profiler { desc: None };
        }
        let allowed = &stack.filter_data.allowed;
        if stack.starts.is_empty() && !allowed.is_empty() && !allowed.contains(desc) {
            return Profiler { desc: None };
        }

        stack.starts.push(Instant::now());
        Profiler { desc: Some(desc.to_string()) }
    })
}

pub struct Profiler {
    desc: Option<String>,
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
    message: String,
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
            Profiler { desc: Some(desc) } => {
                PROFILE_STACK.with(|stack| {
                    let mut stack = stack.borrow_mut();
                    let start = stack.starts.pop().unwrap();
                    let duration = start.elapsed();
                    let level = stack.starts.len();
                    let message = mem::replace(desc, String::new());
                    stack.messages.push(Message { level, duration, message });
                    if level == 0 {
                        let stdout = stderr();
                        let longer_than = stack.filter_data.longer_than;
                        if duration >= longer_than {
                            print(0, &stack.messages, &mut stdout.lock(), longer_than);
                        }
                        stack.messages.clear();
                    }
                });
            }
            Profiler { desc: None } => (),
        }
    }
}

fn print(lvl: usize, msgs: &[Message], out: &mut impl Write, longer_than: Duration) {
    let mut last = 0;
    let indent = repeat("    ").take(lvl + 1).collect::<String>();
    // We output hierarchy for long calls, but sum up all short calls
    let mut short = Vec::new();
    for (i, &Message { level, duration, message: ref msg }) in msgs.iter().enumerate() {
        if level != lvl {
            continue;
        }
        if duration >= longer_than {
            writeln!(out, "{} {:6}ms - {}", indent, duration.as_millis(), msg)
                .expect("printing profiling info to stdout");

            print(lvl + 1, &msgs[last..i], out, longer_than);
        } else {
            short.push((msg, duration))
        }

        last = i;
    }
    short.sort_by_key(|(msg, _time)| *msg);
    for (msg, entires) in short.iter().group_by(|(msg, _time)| msg).into_iter() {
        let mut count = 0;
        let mut total_duration = Duration::default();
        entires.for_each(|(_msg, time)| {
            count += 1;
            total_duration += *time;
        });
        writeln!(out, "{} {:6}ms - {} ({} calls)", indent, total_duration.as_millis(), msg, count)
            .expect("printing profiling info to stdout");
    }
}

/// Prints backtrace to stderr, useful for debugging.
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
/// 5. Use something like `pprof -svg target/release/ra_cli ./out.profile` to see the results.
///
/// For example, here's how I run profiling on NixOS:
///
/// ```bash
/// $ nix-shell -p gperftools --run \
///     'cargo run --release -p ra_cli -- parse < ~/projects/rustbench/parser.rs > /dev/null'
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
}
