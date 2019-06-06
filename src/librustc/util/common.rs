#![allow(non_camel_case_types)]

use rustc_data_structures::{fx::FxHashMap, sync::Lock};

use std::cell::{RefCell, Cell};
use std::fmt::Debug;
use std::hash::Hash;
use std::panic;
use std::env;
use std::time::{Duration, Instant};

use std::sync::mpsc::{Sender};
use syntax_pos::{SpanData};
use syntax::symbol::{Symbol, sym};
use rustc_macros::HashStable;
use crate::ty::TyCtxt;
use crate::dep_graph::{DepNode};
use lazy_static;
use crate::session::Session;

// The name of the associated type for `Fn` return types.
pub const FN_OUTPUT_NAME: Symbol = sym::Output;

// Useful type to use with `Result<>` indicate that an error has already
// been reported to the user, so no need to continue checking.
#[derive(Clone, Copy, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct ErrorReported;

thread_local!(static TIME_DEPTH: Cell<usize> = Cell::new(0));

lazy_static! {
    static ref DEFAULT_HOOK: Box<dyn Fn(&panic::PanicInfo<'_>) + Sync + Send + 'static> = {
        let hook = panic::take_hook();
        panic::set_hook(Box::new(panic_hook));
        hook
    };
}

fn panic_hook(info: &panic::PanicInfo<'_>) {
    (*DEFAULT_HOOK)(info);

    let backtrace = env::var_os("RUST_BACKTRACE").map(|x| &x != "0").unwrap_or(false);

    if backtrace {
        TyCtxt::try_print_query_stack();
    }

    #[cfg(windows)]
    unsafe {
        if env::var("RUSTC_BREAK_ON_ICE").is_ok() {
            extern "system" {
                fn DebugBreak();
            }
            // Trigger a debugger if we crashed during bootstrap.
            DebugBreak();
        }
    }
}

pub fn install_panic_hook() {
    lazy_static::initialize(&DEFAULT_HOOK);
}

/// Parameters to the `Dump` variant of type `ProfileQueriesMsg`.
#[derive(Clone,Debug)]
pub struct ProfQDumpParams {
    /// A base path for the files we will dump.
    pub path:String,
    /// To ensure that the compiler waits for us to finish our dumps.
    pub ack:Sender<()>,
    /// Toggle dumping a log file with every `ProfileQueriesMsg`.
    pub dump_profq_msg_log:bool,
}

#[allow(nonstandard_style)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryMsg {
    pub query: &'static str,
    pub msg: Option<String>,
}

/// A sequence of these messages induce a trace of query-based incremental compilation.
// FIXME(matthewhammer): Determine whether we should include cycle detection here or not.
#[derive(Clone,Debug)]
pub enum ProfileQueriesMsg {
    /// Begin a timed pass.
    TimeBegin(String),
    /// End a timed pass.
    TimeEnd,
    /// Begin a task (see `dep_graph::graph::with_task`).
    TaskBegin(DepNode),
    /// End a task.
    TaskEnd,
    /// Begin a new query.
    /// Cannot use `Span` because queries are sent to other thread.
    QueryBegin(SpanData, QueryMsg),
    /// Query is satisfied by using an already-known value for the given key.
    CacheHit,
    /// Query requires running a provider; providers may nest, permitting queries to nest.
    ProviderBegin,
    /// Query is satisfied by a provider terminating with a value.
    ProviderEnd,
    /// Dump a record of the queries to the given path.
    Dump(ProfQDumpParams),
    /// Halt the profiling/monitoring background thread.
    Halt
}

/// If enabled, send a message to the profile-queries thread.
pub fn profq_msg(sess: &Session, msg: ProfileQueriesMsg) {
    if let Some(s) = sess.profile_channel.borrow().as_ref() {
        s.send(msg).unwrap()
    } else {
        // Do nothing.
    }
}

/// Set channel for profile queries channel.
pub fn profq_set_chan(sess: &Session, s: Sender<ProfileQueriesMsg>) -> bool {
    let mut channel = sess.profile_channel.borrow_mut();
    if channel.is_none() {
        *channel = Some(s);
        true
    } else {
        false
    }
}

/// Read the current depth of `time()` calls. This is used to
/// encourage indentation across threads.
pub fn time_depth() -> usize {
    TIME_DEPTH.with(|slot| slot.get())
}

/// Sets the current depth of `time()` calls. The idea is to call
/// `set_time_depth()` with the result from `time_depth()` in the
/// parent thread.
pub fn set_time_depth(depth: usize) {
    TIME_DEPTH.with(|slot| slot.set(depth));
}

pub fn time<T, F>(sess: &Session, what: &str, f: F) -> T where
    F: FnOnce() -> T,
{
    time_ext(sess.time_passes(), Some(sess), what, f)
}

pub fn time_ext<T, F>(do_it: bool, sess: Option<&Session>, what: &str, f: F) -> T where
    F: FnOnce() -> T,
{
    if !do_it { return f(); }

    let old = TIME_DEPTH.with(|slot| {
        let r = slot.get();
        slot.set(r + 1);
        r
    });

    if let Some(sess) = sess {
        if cfg!(debug_assertions) {
            profq_msg(sess, ProfileQueriesMsg::TimeBegin(what.to_string()))
        }
    }
    let start = Instant::now();
    let rv = f();
    let dur = start.elapsed();
    if let Some(sess) = sess {
        if cfg!(debug_assertions) {
            profq_msg(sess, ProfileQueriesMsg::TimeEnd)
        }
    }

    print_time_passes_entry_internal(what, dur);

    TIME_DEPTH.with(|slot| slot.set(old));

    rv
}

pub fn print_time_passes_entry(do_it: bool, what: &str, dur: Duration) {
    if !do_it {
        return
    }

    let old = TIME_DEPTH.with(|slot| {
        let r = slot.get();
        slot.set(r + 1);
        r
    });

    print_time_passes_entry_internal(what, dur);

    TIME_DEPTH.with(|slot| slot.set(old));
}

fn print_time_passes_entry_internal(what: &str, dur: Duration) {
    let indentation = TIME_DEPTH.with(|slot| slot.get());

    let mem_string = match get_resident() {
        Some(n) => {
            let mb = n as f64 / 1_000_000.0;
            format!("; rss: {}MB", mb.round() as usize)
        }
        None => String::new(),
    };
    println!("{}time: {}{}\t{}",
             "  ".repeat(indentation),
             duration_to_secs_str(dur),
             mem_string,
             what);
}

// Hack up our own formatting for the duration to make it easier for scripts
// to parse (always use the same number of decimal places and the same unit).
pub fn duration_to_secs_str(dur: Duration) -> String {
    const NANOS_PER_SEC: f64 = 1_000_000_000.0;
    let secs = dur.as_secs() as f64 +
               dur.subsec_nanos() as f64 / NANOS_PER_SEC;

    format!("{:.3}", secs)
}

pub fn to_readable_str(mut val: usize) -> String {
    let mut groups = vec![];
    loop {
        let group = val % 1000;

        val /= 1000;

        if val == 0 {
            groups.push(group.to_string());
            break;
        } else {
            groups.push(format!("{:03}", group));
        }
    }

    groups.reverse();

    groups.join("_")
}

pub fn record_time<T, F>(accu: &Lock<Duration>, f: F) -> T where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let rv = f();
    let duration = start.elapsed();
    let mut accu = accu.lock();
    *accu = *accu + duration;
    rv
}

// Memory reporting
#[cfg(unix)]
fn get_resident() -> Option<usize> {
    use std::fs;

    let field = 1;
    let contents = fs::read("/proc/self/statm").ok()?;
    let contents = String::from_utf8(contents).ok()?;
    let s = contents.split_whitespace().nth(field)?;
    let npages = s.parse::<usize>().ok()?;
    Some(npages * 4096)
}

#[cfg(windows)]
fn get_resident() -> Option<usize> {
    type BOOL = i32;
    type DWORD = u32;
    type HANDLE = *mut u8;
    use libc::size_t;
    use std::mem;
    #[repr(C)]
    #[allow(non_snake_case)]
    struct PROCESS_MEMORY_COUNTERS {
        cb: DWORD,
        PageFaultCount: DWORD,
        PeakWorkingSetSize: size_t,
        WorkingSetSize: size_t,
        QuotaPeakPagedPoolUsage: size_t,
        QuotaPagedPoolUsage: size_t,
        QuotaPeakNonPagedPoolUsage: size_t,
        QuotaNonPagedPoolUsage: size_t,
        PagefileUsage: size_t,
        PeakPagefileUsage: size_t,
    }
    type PPROCESS_MEMORY_COUNTERS = *mut PROCESS_MEMORY_COUNTERS;
    #[link(name = "psapi")]
    extern "system" {
        fn GetCurrentProcess() -> HANDLE;
        fn GetProcessMemoryInfo(Process: HANDLE,
                                ppsmemCounters: PPROCESS_MEMORY_COUNTERS,
                                cb: DWORD) -> BOOL;
    }
    let mut pmc: PROCESS_MEMORY_COUNTERS = unsafe { mem::zeroed() };
    pmc.cb = mem::size_of_val(&pmc) as DWORD;
    match unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) } {
        0 => None,
        _ => Some(pmc.WorkingSetSize as usize),
    }
}

pub fn indent<R, F>(op: F) -> R where
    R: Debug,
    F: FnOnce() -> R,
{
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r = op();
    debug!("<< (Result = {:?})", r);
    r
}

pub struct Indenter {
    _cannot_construct_outside_of_this_module: (),
}

impl Drop for Indenter {
    fn drop(&mut self) { debug!("<<"); }
}

pub fn indenter() -> Indenter {
    debug!(">>");
    Indenter { _cannot_construct_outside_of_this_module: () }
}

pub trait MemoizationMap {
    type Key: Clone;
    type Value: Clone;

    /// If `key` is present in the map, return the value,
    /// otherwise invoke `op` and store the value in the map.
    ///
    /// N.B., if the receiver is a `DepTrackingMap`, special care is
    /// needed in the `op` to ensure that the correct edges are
    /// added into the dep graph. See the `DepTrackingMap` impl for
    /// more details!
    fn memoize<OP>(&self, key: Self::Key, op: OP) -> Self::Value
        where OP: FnOnce() -> Self::Value;
}

impl<K, V> MemoizationMap for RefCell<FxHashMap<K,V>>
    where K: Hash+Eq+Clone, V: Clone
{
    type Key = K;
    type Value = V;

    fn memoize<OP>(&self, key: K, op: OP) -> V
        where OP: FnOnce() -> V
    {
        let result = self.borrow().get(&key).cloned();
        match result {
            Some(result) => result,
            None => {
                let result = op();
                self.borrow_mut().insert(key, result.clone());
                result
            }
        }
    }
}

#[test]
fn test_to_readable_str() {
    assert_eq!("0", to_readable_str(0));
    assert_eq!("1", to_readable_str(1));
    assert_eq!("99", to_readable_str(99));
    assert_eq!("999", to_readable_str(999));
    assert_eq!("1_000", to_readable_str(1_000));
    assert_eq!("1_001", to_readable_str(1_001));
    assert_eq!("999_999", to_readable_str(999_999));
    assert_eq!("1_000_000", to_readable_str(1_000_000));
    assert_eq!("1_234_567", to_readable_str(1_234_567));
}
