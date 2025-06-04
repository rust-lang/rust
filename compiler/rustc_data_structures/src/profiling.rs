//! # Rust Compiler Self-Profiling
//!
//! This module implements the basic framework for the compiler's self-
//! profiling support. It provides the `SelfProfiler` type which enables
//! recording "events". An event is something that starts and ends at a given
//! point in time and has an ID and a kind attached to it. This allows for
//! tracing the compiler's activity.
//!
//! Internally this module uses the custom tailored [measureme][mm] crate for
//! efficiently recording events to disk in a compact format that can be
//! post-processed and analyzed by the suite of tools in the `measureme`
//! project. The highest priority for the tracing framework is on incurring as
//! little overhead as possible.
//!
//!
//! ## Event Overview
//!
//! Events have a few properties:
//!
//! - The `event_kind` designates the broad category of an event (e.g. does it
//!   correspond to the execution of a query provider or to loading something
//!   from the incr. comp. on-disk cache, etc).
//! - The `event_id` designates the query invocation or function call it
//!   corresponds to, possibly including the query key or function arguments.
//! - Each event stores the ID of the thread it was recorded on.
//! - The timestamp stores beginning and end of the event, or the single point
//!   in time it occurred at for "instant" events.
//!
//!
//! ## Event Filtering
//!
//! Event generation can be filtered by event kind. Recording all possible
//! events generates a lot of data, much of which is not needed for most kinds
//! of analysis. So, in order to keep overhead as low as possible for a given
//! use case, the `SelfProfiler` will only record the kinds of events that
//! pass the filter specified as a command line argument to the compiler.
//!
//!
//! ## `event_id` Assignment
//!
//! As far as `measureme` is concerned, `event_id`s are just strings. However,
//! it would incur too much overhead to generate and persist each `event_id`
//! string at the point where the event is recorded. In order to make this more
//! efficient `measureme` has two features:
//!
//! - Strings can share their content, so that re-occurring parts don't have to
//!   be copied over and over again. One allocates a string in `measureme` and
//!   gets back a `StringId`. This `StringId` is then used to refer to that
//!   string. `measureme` strings are actually DAGs of string components so that
//!   arbitrary sharing of substrings can be done efficiently. This is useful
//!   because `event_id`s contain lots of redundant text like query names or
//!   def-path components.
//!
//! - `StringId`s can be "virtual" which means that the client picks a numeric
//!   ID according to some application-specific scheme and can later make that
//!   ID be mapped to an actual string. This is used to cheaply generate
//!   `event_id`s while the events actually occur, causing little timing
//!   distortion, and then later map those `StringId`s, in bulk, to actual
//!   `event_id` strings. This way the largest part of the tracing overhead is
//!   localized to one contiguous chunk of time.
//!
//! How are these `event_id`s generated in the compiler? For things that occur
//! infrequently (e.g. "generic activities"), we just allocate the string the
//! first time it is used and then keep the `StringId` in a hash table. This
//! is implemented in `SelfProfiler::get_or_alloc_cached_string()`.
//!
//! For queries it gets more interesting: First we need a unique numeric ID for
//! each query invocation (the `QueryInvocationId`). This ID is used as the
//! virtual `StringId` we use as `event_id` for a given event. This ID has to
//! be available both when the query is executed and later, together with the
//! query key, when we allocate the actual `event_id` strings in bulk.
//!
//! We could make the compiler generate and keep track of such an ID for each
//! query invocation but luckily we already have something that fits all the
//! the requirements: the query's `DepNodeIndex`. So we use the numeric value
//! of the `DepNodeIndex` as `event_id` when recording the event and then,
//! just before the query context is dropped, we walk the entire query cache
//! (which stores the `DepNodeIndex` along with the query key for each
//! invocation) and allocate the corresponding strings together with a mapping
//! for `DepNodeIndex as StringId`.
//!
//! [mm]: https://github.com/rust-lang/measureme/

use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::error::Error;
use std::fmt::Display;
use std::intrinsics::unlikely;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{fs, process};

pub use measureme::EventId;
use measureme::{EventIdBuilder, Profiler, SerializableString, StringId};
use parking_lot::RwLock;
use smallvec::SmallVec;
use tracing::warn;

use crate::fx::FxHashMap;
use crate::outline;

bitflags::bitflags! {
    #[derive(Clone, Copy)]
    struct EventFilter: u16 {
        const GENERIC_ACTIVITIES  = 1 << 0;
        const QUERY_PROVIDERS     = 1 << 1;
        const QUERY_CACHE_HITS    = 1 << 2;
        const QUERY_BLOCKED       = 1 << 3;
        const INCR_CACHE_LOADS    = 1 << 4;

        const QUERY_KEYS          = 1 << 5;
        const FUNCTION_ARGS       = 1 << 6;
        const LLVM                = 1 << 7;
        const INCR_RESULT_HASHING = 1 << 8;
        const ARTIFACT_SIZES = 1 << 9;

        const DEFAULT = Self::GENERIC_ACTIVITIES.bits() |
                        Self::QUERY_PROVIDERS.bits() |
                        Self::QUERY_BLOCKED.bits() |
                        Self::INCR_CACHE_LOADS.bits() |
                        Self::INCR_RESULT_HASHING.bits() |
                        Self::ARTIFACT_SIZES.bits();

        const ARGS = Self::QUERY_KEYS.bits() | Self::FUNCTION_ARGS.bits();
    }
}

// keep this in sync with the `-Z self-profile-events` help message in rustc_session/options.rs
const EVENT_FILTERS_BY_NAME: &[(&str, EventFilter)] = &[
    ("none", EventFilter::empty()),
    ("all", EventFilter::all()),
    ("default", EventFilter::DEFAULT),
    ("generic-activity", EventFilter::GENERIC_ACTIVITIES),
    ("query-provider", EventFilter::QUERY_PROVIDERS),
    ("query-cache-hit", EventFilter::QUERY_CACHE_HITS),
    ("query-blocked", EventFilter::QUERY_BLOCKED),
    ("incr-cache-load", EventFilter::INCR_CACHE_LOADS),
    ("query-keys", EventFilter::QUERY_KEYS),
    ("function-args", EventFilter::FUNCTION_ARGS),
    ("args", EventFilter::ARGS),
    ("llvm", EventFilter::LLVM),
    ("incr-result-hashing", EventFilter::INCR_RESULT_HASHING),
    ("artifact-sizes", EventFilter::ARTIFACT_SIZES),
];

/// Something that uniquely identifies a query invocation.
pub struct QueryInvocationId(pub u32);

/// Which format to use for `-Z time-passes`
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum TimePassesFormat {
    /// Emit human readable text
    Text,
    /// Emit structured JSON
    Json,
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

    // Print verbose generic activities to stderr.
    print_verbose_generic_activities: Option<TimePassesFormat>,
}

impl SelfProfilerRef {
    pub fn new(
        profiler: Option<Arc<SelfProfiler>>,
        print_verbose_generic_activities: Option<TimePassesFormat>,
    ) -> SelfProfilerRef {
        // If there is no SelfProfiler then the filter mask is set to NONE,
        // ensuring that nothing ever tries to actually access it.
        let event_filter_mask =
            profiler.as_ref().map_or(EventFilter::empty(), |p| p.event_filter_mask);

        SelfProfilerRef { profiler, event_filter_mask, print_verbose_generic_activities }
    }

    /// This shim makes sure that calls only get executed if the filter mask
    /// lets them pass. It also contains some trickery to make sure that
    /// code is optimized for non-profiling compilation sessions, i.e. anything
    /// past the filter check is never inlined so it doesn't clutter the fast
    /// path.
    #[inline(always)]
    fn exec<F>(&self, event_filter: EventFilter, f: F) -> TimingGuard<'_>
    where
        F: for<'a> FnOnce(&'a SelfProfiler) -> TimingGuard<'a>,
    {
        #[inline(never)]
        #[cold]
        fn cold_call<F>(profiler_ref: &SelfProfilerRef, f: F) -> TimingGuard<'_>
        where
            F: for<'a> FnOnce(&'a SelfProfiler) -> TimingGuard<'a>,
        {
            let profiler = profiler_ref.profiler.as_ref().unwrap();
            f(profiler)
        }

        if self.event_filter_mask.contains(event_filter) {
            cold_call(self, f)
        } else {
            TimingGuard::none()
        }
    }

    /// Start profiling a verbose generic activity. Profiling continues until the
    /// VerboseTimingGuard returned from this call is dropped. In addition to recording
    /// a measureme event, "verbose" generic activities also print a timing entry to
    /// stderr if the compiler is invoked with -Ztime-passes.
    pub fn verbose_generic_activity(&self, event_label: &'static str) -> VerboseTimingGuard<'_> {
        let message_and_format =
            self.print_verbose_generic_activities.map(|format| (event_label.to_owned(), format));

        VerboseTimingGuard::start(message_and_format, self.generic_activity(event_label))
    }

    /// Like `verbose_generic_activity`, but with an extra arg.
    pub fn verbose_generic_activity_with_arg<A>(
        &self,
        event_label: &'static str,
        event_arg: A,
    ) -> VerboseTimingGuard<'_>
    where
        A: Borrow<str> + Into<String>,
    {
        let message_and_format = self
            .print_verbose_generic_activities
            .map(|format| (format!("{}({})", event_label, event_arg.borrow()), format));

        VerboseTimingGuard::start(
            message_and_format,
            self.generic_activity_with_arg(event_label, event_arg),
        )
    }

    /// Start profiling a generic activity. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn generic_activity(&self, event_label: &'static str) -> TimingGuard<'_> {
        self.exec(EventFilter::GENERIC_ACTIVITIES, |profiler| {
            let event_label = profiler.get_or_alloc_cached_string(event_label);
            let event_id = EventId::from_label(event_label);
            TimingGuard::start(profiler, profiler.generic_activity_event_kind, event_id)
        })
    }

    /// Start profiling with some event filter for a given event. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn generic_activity_with_event_id(&self, event_id: EventId) -> TimingGuard<'_> {
        self.exec(EventFilter::GENERIC_ACTIVITIES, |profiler| {
            TimingGuard::start(profiler, profiler.generic_activity_event_kind, event_id)
        })
    }

    /// Start profiling a generic activity. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn generic_activity_with_arg<A>(
        &self,
        event_label: &'static str,
        event_arg: A,
    ) -> TimingGuard<'_>
    where
        A: Borrow<str> + Into<String>,
    {
        self.exec(EventFilter::GENERIC_ACTIVITIES, |profiler| {
            let builder = EventIdBuilder::new(&profiler.profiler);
            let event_label = profiler.get_or_alloc_cached_string(event_label);
            let event_id = if profiler.event_filter_mask.contains(EventFilter::FUNCTION_ARGS) {
                let event_arg = profiler.get_or_alloc_cached_string(event_arg);
                builder.from_label_and_arg(event_label, event_arg)
            } else {
                builder.from_label(event_label)
            };
            TimingGuard::start(profiler, profiler.generic_activity_event_kind, event_id)
        })
    }

    /// Start profiling a generic activity, allowing costly arguments to be recorded. Profiling
    /// continues until the `TimingGuard` returned from this call is dropped.
    ///
    /// If the arguments to a generic activity are cheap to create, use `generic_activity_with_arg`
    /// or `generic_activity_with_args` for their simpler API. However, if they are costly or
    /// require allocation in sufficiently hot contexts, then this allows for a closure to be called
    /// only when arguments were asked to be recorded via `-Z self-profile-events=args`.
    ///
    /// In this case, the closure will be passed a `&mut EventArgRecorder`, to help with recording
    /// one or many arguments within the generic activity being profiled, by calling its
    /// `record_arg` method for example.
    ///
    /// This `EventArgRecorder` may implement more specific traits from other rustc crates, e.g. for
    /// richer handling of rustc-specific argument types, while keeping this single entry-point API
    /// for recording arguments.
    ///
    /// Note: recording at least one argument is *required* for the self-profiler to create the
    /// `TimingGuard`. A panic will be triggered if that doesn't happen. This function exists
    /// explicitly to record arguments, so it fails loudly when there are none to record.
    ///
    #[inline(always)]
    pub fn generic_activity_with_arg_recorder<F>(
        &self,
        event_label: &'static str,
        mut f: F,
    ) -> TimingGuard<'_>
    where
        F: FnMut(&mut EventArgRecorder<'_>),
    {
        // Ensure this event will only be recorded when self-profiling is turned on.
        self.exec(EventFilter::GENERIC_ACTIVITIES, |profiler| {
            let builder = EventIdBuilder::new(&profiler.profiler);
            let event_label = profiler.get_or_alloc_cached_string(event_label);

            // Ensure the closure to create event arguments will only be called when argument
            // recording is turned on.
            let event_id = if profiler.event_filter_mask.contains(EventFilter::FUNCTION_ARGS) {
                // Set up the builder and call the user-provided closure to record potentially
                // costly event arguments.
                let mut recorder = EventArgRecorder { profiler, args: SmallVec::new() };
                f(&mut recorder);

                // It is expected that the closure will record at least one argument. If that
                // doesn't happen, it's a bug: we've been explicitly called in order to record
                // arguments, so we fail loudly when there are none to record.
                if recorder.args.is_empty() {
                    panic!(
                        "The closure passed to `generic_activity_with_arg_recorder` needs to \
                         record at least one argument"
                    );
                }

                builder.from_label_and_args(event_label, &recorder.args)
            } else {
                builder.from_label(event_label)
            };
            TimingGuard::start(profiler, profiler.generic_activity_event_kind, event_id)
        })
    }

    /// Record the size of an artifact that the compiler produces
    ///
    /// `artifact_kind` is the class of artifact (e.g., query_cache, object_file, etc.)
    /// `artifact_name` is an identifier to the specific artifact being stored (usually a filename)
    #[inline(always)]
    pub fn artifact_size<A>(&self, artifact_kind: &str, artifact_name: A, size: u64)
    where
        A: Borrow<str> + Into<String>,
    {
        drop(self.exec(EventFilter::ARTIFACT_SIZES, |profiler| {
            let builder = EventIdBuilder::new(&profiler.profiler);
            let event_label = profiler.get_or_alloc_cached_string(artifact_kind);
            let event_arg = profiler.get_or_alloc_cached_string(artifact_name);
            let event_id = builder.from_label_and_arg(event_label, event_arg);
            let thread_id = get_thread_id();

            profiler.profiler.record_integer_event(
                profiler.artifact_size_event_kind,
                event_id,
                thread_id,
                size,
            );

            TimingGuard::none()
        }))
    }

    #[inline(always)]
    pub fn generic_activity_with_args(
        &self,
        event_label: &'static str,
        event_args: &[String],
    ) -> TimingGuard<'_> {
        self.exec(EventFilter::GENERIC_ACTIVITIES, |profiler| {
            let builder = EventIdBuilder::new(&profiler.profiler);
            let event_label = profiler.get_or_alloc_cached_string(event_label);
            let event_id = if profiler.event_filter_mask.contains(EventFilter::FUNCTION_ARGS) {
                let event_args: Vec<_> = event_args
                    .iter()
                    .map(|s| profiler.get_or_alloc_cached_string(&s[..]))
                    .collect();
                builder.from_label_and_args(event_label, &event_args)
            } else {
                builder.from_label(event_label)
            };
            TimingGuard::start(profiler, profiler.generic_activity_event_kind, event_id)
        })
    }

    /// Start profiling a query provider. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn query_provider(&self) -> TimingGuard<'_> {
        self.exec(EventFilter::QUERY_PROVIDERS, |profiler| {
            TimingGuard::start(profiler, profiler.query_event_kind, EventId::INVALID)
        })
    }

    /// Record a query in-memory cache hit.
    #[inline(always)]
    pub fn query_cache_hit(&self, query_invocation_id: QueryInvocationId) {
        #[inline(never)]
        #[cold]
        fn cold_call(profiler_ref: &SelfProfilerRef, query_invocation_id: QueryInvocationId) {
            profiler_ref.instant_query_event(
                |profiler| profiler.query_cache_hit_event_kind,
                query_invocation_id,
            );
        }

        if unlikely(self.event_filter_mask.contains(EventFilter::QUERY_CACHE_HITS)) {
            cold_call(self, query_invocation_id);
        }
    }

    /// Start profiling a query being blocked on a concurrent execution.
    /// Profiling continues until the TimingGuard returned from this call is
    /// dropped.
    #[inline(always)]
    pub fn query_blocked(&self) -> TimingGuard<'_> {
        self.exec(EventFilter::QUERY_BLOCKED, |profiler| {
            TimingGuard::start(profiler, profiler.query_blocked_event_kind, EventId::INVALID)
        })
    }

    /// Start profiling how long it takes to load a query result from the
    /// incremental compilation on-disk cache. Profiling continues until the
    /// TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn incr_cache_loading(&self) -> TimingGuard<'_> {
        self.exec(EventFilter::INCR_CACHE_LOADS, |profiler| {
            TimingGuard::start(
                profiler,
                profiler.incremental_load_result_event_kind,
                EventId::INVALID,
            )
        })
    }

    /// Start profiling how long it takes to hash query results for incremental compilation.
    /// Profiling continues until the TimingGuard returned from this call is dropped.
    #[inline(always)]
    pub fn incr_result_hashing(&self) -> TimingGuard<'_> {
        self.exec(EventFilter::INCR_RESULT_HASHING, |profiler| {
            TimingGuard::start(
                profiler,
                profiler.incremental_result_hashing_event_kind,
                EventId::INVALID,
            )
        })
    }

    #[inline(always)]
    fn instant_query_event(
        &self,
        event_kind: fn(&SelfProfiler) -> StringId,
        query_invocation_id: QueryInvocationId,
    ) {
        let event_id = StringId::new_virtual(query_invocation_id.0);
        let thread_id = get_thread_id();
        let profiler = self.profiler.as_ref().unwrap();
        profiler.profiler.record_instant_event(
            event_kind(profiler),
            EventId::from_virtual(event_id),
            thread_id,
        );
    }

    pub fn with_profiler(&self, f: impl FnOnce(&SelfProfiler)) {
        if let Some(profiler) = &self.profiler {
            f(profiler)
        }
    }

    /// Gets a `StringId` for the given string. This method makes sure that
    /// any strings going through it will only be allocated once in the
    /// profiling data.
    /// Returns `None` if the self-profiling is not enabled.
    pub fn get_or_alloc_cached_string(&self, s: &str) -> Option<StringId> {
        self.profiler.as_ref().map(|p| p.get_or_alloc_cached_string(s))
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.profiler.is_some()
    }

    #[inline]
    pub fn llvm_recording_enabled(&self) -> bool {
        self.event_filter_mask.contains(EventFilter::LLVM)
    }
    #[inline]
    pub fn get_self_profiler(&self) -> Option<Arc<SelfProfiler>> {
        self.profiler.clone()
    }
}

/// A helper for recording costly arguments to self-profiling events. Used with
/// `SelfProfilerRef::generic_activity_with_arg_recorder`.
pub struct EventArgRecorder<'p> {
    /// The `SelfProfiler` used to intern the event arguments that users will ask to record.
    profiler: &'p SelfProfiler,

    /// The interned event arguments to be recorded in the generic activity event.
    ///
    /// The most common case, when actually recording event arguments, is to have one argument. Then
    /// followed by recording two, in a couple places.
    args: SmallVec<[StringId; 2]>,
}

impl EventArgRecorder<'_> {
    /// Records a single argument within the current generic activity being profiled.
    ///
    /// Note: when self-profiling with costly event arguments, at least one argument
    /// needs to be recorded. A panic will be triggered if that doesn't happen.
    pub fn record_arg<A>(&mut self, event_arg: A)
    where
        A: Borrow<str> + Into<String>,
    {
        let event_arg = self.profiler.get_or_alloc_cached_string(event_arg);
        self.args.push(event_arg);
    }
}

pub struct SelfProfiler {
    profiler: Profiler,
    event_filter_mask: EventFilter,

    string_cache: RwLock<FxHashMap<String, StringId>>,

    query_event_kind: StringId,
    generic_activity_event_kind: StringId,
    incremental_load_result_event_kind: StringId,
    incremental_result_hashing_event_kind: StringId,
    query_blocked_event_kind: StringId,
    query_cache_hit_event_kind: StringId,
    artifact_size_event_kind: StringId,
}

impl SelfProfiler {
    pub fn new(
        output_directory: &Path,
        crate_name: Option<&str>,
        event_filters: Option<&[String]>,
        counter_name: &str,
    ) -> Result<SelfProfiler, Box<dyn Error + Send + Sync>> {
        fs::create_dir_all(output_directory)?;

        let crate_name = crate_name.unwrap_or("unknown-crate");
        // HACK(eddyb) we need to pad the PID, strange as it may seem, as its
        // length can behave as a source of entropy for heap addresses, when
        // ASLR is disabled and the heap is otherwise deterministic.
        let pid: u32 = process::id();
        let filename = format!("{crate_name}-{pid:07}.rustc_profile");
        let path = output_directory.join(filename);
        let profiler =
            Profiler::with_counter(&path, measureme::counters::Counter::by_name(counter_name)?)?;

        let query_event_kind = profiler.alloc_string("Query");
        let generic_activity_event_kind = profiler.alloc_string("GenericActivity");
        let incremental_load_result_event_kind = profiler.alloc_string("IncrementalLoadResult");
        let incremental_result_hashing_event_kind =
            profiler.alloc_string("IncrementalResultHashing");
        let query_blocked_event_kind = profiler.alloc_string("QueryBlocked");
        let query_cache_hit_event_kind = profiler.alloc_string("QueryCacheHit");
        let artifact_size_event_kind = profiler.alloc_string("ArtifactSize");

        let mut event_filter_mask = EventFilter::empty();

        if let Some(event_filters) = event_filters {
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
            if !unknown_events.is_empty() {
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
            string_cache: RwLock::new(FxHashMap::default()),
            query_event_kind,
            generic_activity_event_kind,
            incremental_load_result_event_kind,
            incremental_result_hashing_event_kind,
            query_blocked_event_kind,
            query_cache_hit_event_kind,
            artifact_size_event_kind,
        })
    }

    /// Allocates a new string in the profiling data. Does not do any caching
    /// or deduplication.
    pub fn alloc_string<STR: SerializableString + ?Sized>(&self, s: &STR) -> StringId {
        self.profiler.alloc_string(s)
    }

    /// Gets a `StringId` for the given string. This method makes sure that
    /// any strings going through it will only be allocated once in the
    /// profiling data.
    pub fn get_or_alloc_cached_string<A>(&self, s: A) -> StringId
    where
        A: Borrow<str> + Into<String>,
    {
        // Only acquire a read-lock first since we assume that the string is
        // already present in the common case.
        {
            let string_cache = self.string_cache.read();

            if let Some(&id) = string_cache.get(s.borrow()) {
                return id;
            }
        }

        let mut string_cache = self.string_cache.write();
        // Check if the string has already been added in the small time window
        // between dropping the read lock and acquiring the write lock.
        match string_cache.entry(s.into()) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let string_id = self.profiler.alloc_string(&e.key()[..]);
                *e.insert(string_id)
            }
        }
    }

    pub fn map_query_invocation_id_to_string(&self, from: QueryInvocationId, to: StringId) {
        let from = StringId::new_virtual(from.0);
        self.profiler.map_virtual_to_concrete_string(from, to);
    }

    pub fn bulk_map_query_invocation_id_to_single_string<I>(&self, from: I, to: StringId)
    where
        I: Iterator<Item = QueryInvocationId> + ExactSizeIterator,
    {
        let from = from.map(|qid| StringId::new_virtual(qid.0));
        self.profiler.bulk_map_virtual_to_single_concrete_string(from, to);
    }

    pub fn query_key_recording_enabled(&self) -> bool {
        self.event_filter_mask.contains(EventFilter::QUERY_KEYS)
    }

    pub fn event_id_builder(&self) -> EventIdBuilder<'_> {
        EventIdBuilder::new(&self.profiler)
    }
}

#[must_use]
pub struct TimingGuard<'a>(Option<measureme::TimingGuard<'a>>);

impl<'a> TimingGuard<'a> {
    #[inline]
    pub fn start(
        profiler: &'a SelfProfiler,
        event_kind: StringId,
        event_id: EventId,
    ) -> TimingGuard<'a> {
        let thread_id = get_thread_id();
        let raw_profiler = &profiler.profiler;
        let timing_guard =
            raw_profiler.start_recording_interval_event(event_kind, event_id, thread_id);
        TimingGuard(Some(timing_guard))
    }

    #[inline]
    pub fn finish_with_query_invocation_id(self, query_invocation_id: QueryInvocationId) {
        if let Some(guard) = self.0 {
            outline(|| {
                let event_id = StringId::new_virtual(query_invocation_id.0);
                let event_id = EventId::from_virtual(event_id);
                guard.finish_with_override_event_id(event_id);
            });
        }
    }

    #[inline]
    pub fn none() -> TimingGuard<'a> {
        TimingGuard(None)
    }

    #[inline(always)]
    pub fn run<R>(self, f: impl FnOnce() -> R) -> R {
        let _timer = self;
        f()
    }
}

struct VerboseInfo {
    start_time: Instant,
    start_rss: Option<usize>,
    message: String,
    format: TimePassesFormat,
}

#[must_use]
pub struct VerboseTimingGuard<'a> {
    info: Option<VerboseInfo>,
    _guard: TimingGuard<'a>,
}

impl<'a> VerboseTimingGuard<'a> {
    pub fn start(
        message_and_format: Option<(String, TimePassesFormat)>,
        _guard: TimingGuard<'a>,
    ) -> Self {
        VerboseTimingGuard {
            _guard,
            info: message_and_format.map(|(message, format)| VerboseInfo {
                start_time: Instant::now(),
                start_rss: get_resident_set_size(),
                message,
                format,
            }),
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
        if let Some(info) = &self.info {
            let end_rss = get_resident_set_size();
            let dur = info.start_time.elapsed();
            print_time_passes_entry(&info.message, dur, info.start_rss, end_rss, info.format);
        }
    }
}

struct JsonTimePassesEntry<'a> {
    pass: &'a str,
    time: f64,
    start_rss: Option<usize>,
    end_rss: Option<usize>,
}

impl Display for JsonTimePassesEntry<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { pass: what, time, start_rss, end_rss } = self;
        write!(f, r#"{{"pass":"{what}","time":{time},"rss_start":"#).unwrap();
        match start_rss {
            Some(rss) => write!(f, "{rss}")?,
            None => write!(f, "null")?,
        }
        write!(f, r#","rss_end":"#)?;
        match end_rss {
            Some(rss) => write!(f, "{rss}")?,
            None => write!(f, "null")?,
        }
        write!(f, "}}")?;
        Ok(())
    }
}

pub fn print_time_passes_entry(
    what: &str,
    dur: Duration,
    start_rss: Option<usize>,
    end_rss: Option<usize>,
    format: TimePassesFormat,
) {
    match format {
        TimePassesFormat::Json => {
            let entry =
                JsonTimePassesEntry { pass: what, time: dur.as_secs_f64(), start_rss, end_rss };

            eprintln!(r#"time: {entry}"#);
            return;
        }
        TimePassesFormat::Text => (),
    }

    // Print the pass if its duration is greater than 5 ms, or it changed the
    // measured RSS.
    let is_notable = || {
        if dur.as_millis() > 5 {
            return true;
        }

        if let (Some(start_rss), Some(end_rss)) = (start_rss, end_rss) {
            let change_rss = end_rss.abs_diff(start_rss);
            if change_rss > 0 {
                return true;
            }
        }

        false
    };
    if !is_notable() {
        return;
    }

    let rss_to_mb = |rss| (rss as f64 / 1_000_000.0).round() as usize;
    let rss_change_to_mb = |rss| (rss as f64 / 1_000_000.0).round() as i128;

    let mem_string = match (start_rss, end_rss) {
        (Some(start_rss), Some(end_rss)) => {
            let change_rss = end_rss as i128 - start_rss as i128;

            format!(
                "; rss: {:>4}MB -> {:>4}MB ({:>+5}MB)",
                rss_to_mb(start_rss),
                rss_to_mb(end_rss),
                rss_change_to_mb(change_rss),
            )
        }
        (Some(start_rss), None) => format!("; rss start: {:>4}MB", rss_to_mb(start_rss)),
        (None, Some(end_rss)) => format!("; rss end: {:>4}MB", rss_to_mb(end_rss)),
        (None, None) => String::new(),
    };

    eprintln!("time: {:>7}{}\t{}", duration_to_secs_str(dur), mem_string, what);
}

// Hack up our own formatting for the duration to make it easier for scripts
// to parse (always use the same number of decimal places and the same unit).
pub fn duration_to_secs_str(dur: std::time::Duration) -> String {
    format!("{:.3}", dur.as_secs_f64())
}

fn get_thread_id() -> u32 {
    std::thread::current().id().as_u64().get() as u32
}

// cfg(bootstrap)
macro_rules! cfg_select_dispatch {
    ($($tokens:tt)*) => {
        #[cfg(bootstrap)]
        cfg_match! { $($tokens)* }

        #[cfg(not(bootstrap))]
        cfg_select! { $($tokens)* }
    };
}

// Memory reporting
cfg_select_dispatch! {
    windows => {
        pub fn get_resident_set_size() -> Option<usize> {
            use windows::{
                Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
                Win32::System::Threading::GetCurrentProcess,
            };

            let mut pmc = PROCESS_MEMORY_COUNTERS::default();
            let pmc_size = size_of_val(&pmc);
            unsafe {
                K32GetProcessMemoryInfo(
                    GetCurrentProcess(),
                    &mut pmc,
                    pmc_size as u32,
                )
            }
            .ok()
            .ok()?;

            Some(pmc.WorkingSetSize)
        }
    }
    target_os = "macos" => {
        pub fn get_resident_set_size() -> Option<usize> {
            use libc::{c_int, c_void, getpid, proc_pidinfo, proc_taskinfo, PROC_PIDTASKINFO};
            use std::mem;
            const PROC_TASKINFO_SIZE: c_int = size_of::<proc_taskinfo>() as c_int;

            unsafe {
                let mut info: proc_taskinfo = mem::zeroed();
                let info_ptr = &mut info as *mut proc_taskinfo as *mut c_void;
                let pid = getpid() as c_int;
                let ret = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, info_ptr, PROC_TASKINFO_SIZE);
                if ret == PROC_TASKINFO_SIZE {
                    Some(info.pti_resident_size as usize)
                } else {
                    None
                }
            }
        }
    }
    unix => {
        pub fn get_resident_set_size() -> Option<usize> {
            let field = 1;
            let contents = fs::read("/proc/self/statm").ok()?;
            let contents = String::from_utf8(contents).ok()?;
            let s = contents.split_whitespace().nth(field)?;
            let npages = s.parse::<usize>().ok()?;
            Some(npages * 4096)
        }
    }
    _ => {
        pub fn get_resident_set_size() -> Option<usize> {
            None
        }
    }
}

#[cfg(test)]
mod tests;
