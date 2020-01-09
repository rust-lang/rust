use std::error::Error;
use std::fs;
use std::mem::{self, Discriminant};
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::{Duration, Instant};
use std::u32;

use measureme::StringId;

/// MmapSerializatioSink is faster on macOS and Linux
/// but FileSerializationSink is faster on Windows
#[cfg(not(windows))]
type SerializationSink = measureme::MmapSerializationSink;
#[cfg(windows)]
type SerializationSink = measureme::FileSerializationSink;

type Profiler = measureme::Profiler<SerializationSink>;

pub trait QueryName: Sized + Copy {
    fn discriminant(self) -> Discriminant<Self>;
    fn as_str(self) -> &'static str;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub enum ProfileCategory {
    Parsing,
    Expansion,
    TypeChecking,
    BorrowChecking,
    Codegen,
    Linking,
    Other,
}

bitflags::bitflags! {
    struct EventFilter: u32 {
        const GENERIC_ACTIVITIES = 1 << 0;
        const QUERY_PROVIDERS    = 1 << 1;
        const QUERY_CACHE_HITS   = 1 << 2;
        const QUERY_BLOCKED      = 1 << 3;
        const INCR_CACHE_LOADS   = 1 << 4;

        const DEFAULT = Self::GENERIC_ACTIVITIES.bits |
                        Self::QUERY_PROVIDERS.bits |
                        Self::QUERY_BLOCKED.bits |
                        Self::INCR_CACHE_LOADS.bits;

        // empty() and none() aren't const-fns unfortunately
        const NONE = 0;
        const ALL  = !Self::NONE.bits;
    }
}

const EVENT_FILTERS_BY_NAME: &[(&str, EventFilter)] = &[
    ("none", EventFilter::NONE),
    ("all", EventFilter::ALL),
    ("generic-activity", EventFilter::GENERIC_ACTIVITIES),
    ("query-provider", EventFilter::QUERY_PROVIDERS),
    ("query-cache-hit", EventFilter::QUERY_CACHE_HITS),
    ("query-blocked", EventFilter::QUERY_BLOCKED),
    ("incr-cache-load", EventFilter::INCR_CACHE_LOADS),
];

fn thread_id_to_u32(tid: ThreadId) -> u32 {
    unsafe { mem::transmute::<ThreadId, u64>(tid) as u32 }
}

/// A reference to the SelfProfiler. It can be cloned and sent across thread
/// boundaries at will.
#[derive(Clone)]
pub struct SelfProfilerRef {
    // This field is `None` if self-profiling is disabled for the current
    // compilation session.
    profiler: Option<Arc<SelfProfiler>>,

    // We store the filter mask directly in the reference because that doesn't
    // cost anything and allows for filtering with checking if the profiler is
    // actually enabled.
    event_filter_mask: EventFilter,

    // Print verbose generic activities to stdout
    print_verbose_generic_activities: bool,

    // Print extra verbose generic activities to stdout
    print_extra_verbose_generic_activities: bool,
}

impl SelfProfilerRef {
    pub fn new(
        profiler: Option<Arc<SelfProfiler>>,
        print_verbose_generic_activities: bool,
        print_extra_verbose_generic_activities: bool,
    ) -> SelfProfilerRef {
        // If there is no SelfProfiler then the filter mask is set to NONE,
        // ensuring that nothing ever tries to actually access it.
        let event_filter_mask =
            profiler.as_ref().map(|p| p.event_filter_mask).unwrap_or(EventFilter::NONE);

        SelfProfilerRef {
            profiler,
            event_filter_mask,
            print_verbose_generic_activities,
            print_extra_verbose_generic_activities,
        }
    }

    // This shim makes sure that calls only get executed if the filter mask
    // lets them pass. It also contains some trickery to make sure that
    // code is optimized for non-profiling compilation sessions, i.e. anything
    // past the filter check is never inlined so it doesn't clutter the fast
    // path.
    #[inline(always)]
    fn exec<F>(&self, event_filter: EventFilter, f: F) -> TimingGuard<'_>
    where
        F: for<'a> FnOnce(&'a SelfProfiler) -> TimingGuard<'a>,
    {
        #[inline(never)]
        fn cold_call<F>(profiler_ref: &SelfProfilerRef, f: F) -> TimingGuard<'_>
        where
            F: for<'a> FnOnce(&'a SelfProfiler) -> TimingGuard<'a>,
        {
            let profiler = profiler_ref.profiler.as_ref().unwrap();
            f(&**profiler)
        }

        if unlikely!(self.event_filter_mask.contains(event_filter)) {
            cold_call(self, f)
        } else {
            TimingGuard::none()
        }
    }

    /// Start profiling a verbose generic activity. Profiling continues until the
    /// VerboseTimingGuard returned from this call is dropped. In addition to recording
    /// a measureme event, "verbose" generic activities also print a timing entry to
    /// stdout if the compiler is invoked with -Ztime or -Ztime-passes.
    #[inline(always)]
    pub fn verbose_generic_activity<'a>(&'a self, event_id: &'a str) -> VerboseTimingGuard<'a> {
        VerboseTimingGuard::start(
            event_id,
            self.print_verbose_generic_activities,
            self.generic_activity(event_id),
        )
    }

    /// Start profiling a extra verbose generic activity. Profiling continues until the
    /// VerboseTimingGuard returned from this call is dropped. In addition to recording
    /// a measureme event, "extra verbose" generic activities also print a timing entry to
    /// stdout if the compiler is invoked with -Ztime-passes.
    #[inline(always)]
    pub fn extra_verbose_generic_activity<'a>(
        &'a self,
        event_id: &'a str,
    ) -> VerboseTimingGuard<'a> {
        // FIXME: This does not yet emit a measureme event
        // because callers encode arguments into `event_id`.
        VerboseTimingGuard::start(
            event_id,
            self.print_extra_verbose_generic_activities,
            TimingGuard::none(),
        )
    }

    /// Start profiling a generic activity. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn generic_activity(&self, event_id: &str) -> TimingGuard<'_> {
        self.exec(EventFilter::GENERIC_ACTIVITIES, |profiler| {
            let event_id = profiler.profiler.alloc_string(event_id);
            TimingGuard::start(profiler, profiler.generic_activity_event_kind, event_id)
        })
    }

    /// Start profiling a query provider. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn query_provider(&self, query_name: impl QueryName) -> TimingGuard<'_> {
        self.exec(EventFilter::QUERY_PROVIDERS, |profiler| {
            let event_id = SelfProfiler::get_query_name_string_id(query_name);
            TimingGuard::start(profiler, profiler.query_event_kind, event_id)
        })
    }

    /// Record a query in-memory cache hit.
    #[inline(always)]
    pub fn query_cache_hit(&self, query_name: impl QueryName) {
        self.instant_query_event(
            |profiler| profiler.query_cache_hit_event_kind,
            query_name,
            EventFilter::QUERY_CACHE_HITS,
        );
    }

    /// Start profiling a query being blocked on a concurrent execution.
    /// Profiling continues until the TimingGuard returned from this call is
    /// dropped.
    #[inline(always)]
    pub fn query_blocked(&self, query_name: impl QueryName) -> TimingGuard<'_> {
        self.exec(EventFilter::QUERY_BLOCKED, |profiler| {
            let event_id = SelfProfiler::get_query_name_string_id(query_name);
            TimingGuard::start(profiler, profiler.query_blocked_event_kind, event_id)
        })
    }

    /// Start profiling how long it takes to load a query result from the
    /// incremental compilation on-disk cache. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn incr_cache_loading(&self, query_name: impl QueryName) -> TimingGuard<'_> {
        self.exec(EventFilter::INCR_CACHE_LOADS, |profiler| {
            let event_id = SelfProfiler::get_query_name_string_id(query_name);
            TimingGuard::start(profiler, profiler.incremental_load_result_event_kind, event_id)
        })
    }

    #[inline(always)]
    fn instant_query_event(
        &self,
        event_kind: fn(&SelfProfiler) -> StringId,
        query_name: impl QueryName,
        event_filter: EventFilter,
    ) {
        drop(self.exec(event_filter, |profiler| {
            let event_id = SelfProfiler::get_query_name_string_id(query_name);
            let thread_id = thread_id_to_u32(std::thread::current().id());

            profiler.profiler.record_instant_event(event_kind(profiler), event_id, thread_id);

            TimingGuard::none()
        }));
    }

    pub fn register_queries(&self, f: impl FnOnce(&SelfProfiler)) {
        if let Some(profiler) = &self.profiler {
            f(&profiler)
        }
    }
}

pub struct SelfProfiler {
    profiler: Profiler,
    event_filter_mask: EventFilter,
    query_event_kind: StringId,
    generic_activity_event_kind: StringId,
    incremental_load_result_event_kind: StringId,
    query_blocked_event_kind: StringId,
    query_cache_hit_event_kind: StringId,
}

impl SelfProfiler {
    pub fn new(
        output_directory: &Path,
        crate_name: Option<&str>,
        event_filters: &Option<Vec<String>>,
    ) -> Result<SelfProfiler, Box<dyn Error>> {
        fs::create_dir_all(output_directory)?;

        let crate_name = crate_name.unwrap_or("unknown-crate");
        let filename = format!("{}-{}.rustc_profile", crate_name, process::id());
        let path = output_directory.join(&filename);
        let profiler = Profiler::new(&path)?;

        let query_event_kind = profiler.alloc_string("Query");
        let generic_activity_event_kind = profiler.alloc_string("GenericActivity");
        let incremental_load_result_event_kind = profiler.alloc_string("IncrementalLoadResult");
        let query_blocked_event_kind = profiler.alloc_string("QueryBlocked");
        let query_cache_hit_event_kind = profiler.alloc_string("QueryCacheHit");

        let mut event_filter_mask = EventFilter::empty();

        if let Some(ref event_filters) = *event_filters {
            let mut unknown_events = vec![];
            for item in event_filters {
                if let Some(&(_, mask)) =
                    EVENT_FILTERS_BY_NAME.iter().find(|&(name, _)| name == item)
                {
                    event_filter_mask |= mask;
                } else {
                    unknown_events.push(item.clone());
                }
            }

            // Warn about any unknown event names
            if unknown_events.len() > 0 {
                unknown_events.sort();
                unknown_events.dedup();

                warn!(
                    "Unknown self-profiler events specified: {}. Available options are: {}.",
                    unknown_events.join(", "),
                    EVENT_FILTERS_BY_NAME
                        .iter()
                        .map(|&(name, _)| name.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        } else {
            event_filter_mask = EventFilter::DEFAULT;
        }

        Ok(SelfProfiler {
            profiler,
            event_filter_mask,
            query_event_kind,
            generic_activity_event_kind,
            incremental_load_result_event_kind,
            query_blocked_event_kind,
            query_cache_hit_event_kind,
        })
    }

    fn get_query_name_string_id(query_name: impl QueryName) -> StringId {
        let discriminant =
            unsafe { mem::transmute::<Discriminant<_>, u64>(query_name.discriminant()) };

        StringId::reserved(discriminant as u32)
    }

    pub fn register_query_name(&self, query_name: impl QueryName) {
        let id = SelfProfiler::get_query_name_string_id(query_name);
        self.profiler.alloc_string_with_reserved_id(id, query_name.as_str());
    }
}

#[must_use]
pub struct TimingGuard<'a>(Option<measureme::TimingGuard<'a, SerializationSink>>);

impl<'a> TimingGuard<'a> {
    #[inline]
    pub fn start(
        profiler: &'a SelfProfiler,
        event_kind: StringId,
        event_id: StringId,
    ) -> TimingGuard<'a> {
        let thread_id = thread_id_to_u32(std::thread::current().id());
        let raw_profiler = &profiler.profiler;
        let timing_guard =
            raw_profiler.start_recording_interval_event(event_kind, event_id, thread_id);
        TimingGuard(Some(timing_guard))
    }

    #[inline]
    pub fn none() -> TimingGuard<'a> {
        TimingGuard(None)
    }
}

#[must_use]
pub struct VerboseTimingGuard<'a> {
    event_id: &'a str,
    start: Option<Instant>,
    _guard: TimingGuard<'a>,
}

impl<'a> VerboseTimingGuard<'a> {
    pub fn start(event_id: &'a str, verbose: bool, _guard: TimingGuard<'a>) -> Self {
        VerboseTimingGuard {
            event_id,
            _guard,
            start: if unlikely!(verbose) { Some(Instant::now()) } else { None },
        }
    }

    #[inline(always)]
    pub fn run<R>(self, f: impl FnOnce() -> R) -> R {
        let _timer = self;
        f()
    }
}

impl Drop for VerboseTimingGuard<'_> {
    fn drop(&mut self) {
        self.start.map(|start| print_time_passes_entry(true, self.event_id, start.elapsed()));
    }
}

pub fn print_time_passes_entry(do_it: bool, what: &str, dur: Duration) {
    if !do_it {
        return;
    }

    let mem_string = match get_resident() {
        Some(n) => {
            let mb = n as f64 / 1_000_000.0;
            format!("; rss: {}MB", mb.round() as usize)
        }
        None => String::new(),
    };
    println!("time: {}{}\t{}", duration_to_secs_str(dur), mem_string, what);
}

// Hack up our own formatting for the duration to make it easier for scripts
// to parse (always use the same number of decimal places and the same unit).
pub fn duration_to_secs_str(dur: std::time::Duration) -> String {
    const NANOS_PER_SEC: f64 = 1_000_000_000.0;
    let secs = dur.as_secs() as f64 + dur.subsec_nanos() as f64 / NANOS_PER_SEC;

    format!("{:.3}", secs)
}

// Memory reporting
#[cfg(unix)]
fn get_resident() -> Option<usize> {
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
    #[allow(non_camel_case_types)]
    type PPROCESS_MEMORY_COUNTERS = *mut PROCESS_MEMORY_COUNTERS;
    #[link(name = "psapi")]
    extern "system" {
        fn GetCurrentProcess() -> HANDLE;
        fn GetProcessMemoryInfo(
            Process: HANDLE,
            ppsmemCounters: PPROCESS_MEMORY_COUNTERS,
            cb: DWORD,
        ) -> BOOL;
    }
    let mut pmc: PROCESS_MEMORY_COUNTERS = unsafe { mem::zeroed() };
    pmc.cb = mem::size_of_val(&pmc) as DWORD;
    match unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) } {
        0 => None,
        _ => Some(pmc.WorkingSetSize as usize),
    }
}
