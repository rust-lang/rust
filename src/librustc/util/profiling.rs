use std::borrow::Cow;
use std::error::Error;
use std::fs;
use std::mem::{self, Discriminant};
use std::path::Path;
use std::process;
use std::thread::ThreadId;
use std::u32;

use crate::ty::query::QueryName;

use measureme::{StringId, TimestampKind};

/// MmapSerializatioSink is faster on macOS and Linux
/// but FileSerializationSink is faster on Windows
#[cfg(not(windows))]
type Profiler = measureme::Profiler<measureme::MmapSerializationSink>;
#[cfg(windows)]
type Profiler = measureme::Profiler<measureme::FileSerializationSink>;

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

bitflags! {
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
    ("query-blocked" , EventFilter::QUERY_BLOCKED),
    ("incr-cache-load", EventFilter::INCR_CACHE_LOADS),
];

fn thread_id_to_u64(tid: ThreadId) -> u64 {
    unsafe { mem::transmute::<ThreadId, u64>(tid) }
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
        event_filters: &Option<Vec<String>>
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
                if let Some(&(_, mask)) = EVENT_FILTERS_BY_NAME.iter()
                                                               .find(|&(name, _)| name == item) {
                    event_filter_mask |= mask;
                } else {
                    unknown_events.push(item.clone());
                }
            }

            // Warn about any unknown event names
            if unknown_events.len() > 0 {
                unknown_events.sort();
                unknown_events.dedup();

                warn!("Unknown self-profiler events specified: {}. Available options are: {}.",
                    unknown_events.join(", "),
                    EVENT_FILTERS_BY_NAME.iter()
                                         .map(|&(name, _)| name.to_string())
                                         .collect::<Vec<_>>()
                                         .join(", "));
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

    fn get_query_name_string_id(query_name: QueryName) -> StringId {
        let discriminant = unsafe {
            mem::transmute::<Discriminant<QueryName>, u64>(mem::discriminant(&query_name))
        };

        StringId::reserved(discriminant as u32)
    }

    pub fn register_query_name(&self, query_name: QueryName) {
        let id = SelfProfiler::get_query_name_string_id(query_name);
        self.profiler.alloc_string_with_reserved_id(id, query_name.as_str());
    }

    #[inline]
    pub fn start_activity(
        &self,
        label: impl Into<Cow<'static, str>>,
    ) {
        if self.event_filter_mask.contains(EventFilter::GENERIC_ACTIVITIES) {
            self.record(&label.into(), self.generic_activity_event_kind, TimestampKind::Start);
        }
    }

    #[inline]
    pub fn end_activity(
        &self,
        label: impl Into<Cow<'static, str>>,
    ) {
        if self.event_filter_mask.contains(EventFilter::GENERIC_ACTIVITIES) {
            self.record(&label.into(), self.generic_activity_event_kind, TimestampKind::End);
        }
    }

    #[inline]
    pub fn record_query_hit(&self, query_name: QueryName) {
        if self.event_filter_mask.contains(EventFilter::QUERY_CACHE_HITS) {
            self.record_query(query_name, self.query_cache_hit_event_kind, TimestampKind::Instant);
        }
    }

    #[inline]
    pub fn start_query(&self, query_name: QueryName) {
        if self.event_filter_mask.contains(EventFilter::QUERY_PROVIDERS) {
            self.record_query(query_name, self.query_event_kind, TimestampKind::Start);
        }
    }

    #[inline]
    pub fn end_query(&self, query_name: QueryName) {
        if self.event_filter_mask.contains(EventFilter::QUERY_PROVIDERS) {
            self.record_query(query_name, self.query_event_kind, TimestampKind::End);
        }
    }

    #[inline]
    pub fn incremental_load_result_start(&self, query_name: QueryName) {
        if self.event_filter_mask.contains(EventFilter::INCR_CACHE_LOADS) {
            self.record_query(
                query_name,
                self.incremental_load_result_event_kind,
                TimestampKind::Start
            );
        }
    }

    #[inline]
    pub fn incremental_load_result_end(&self, query_name: QueryName) {
        if self.event_filter_mask.contains(EventFilter::INCR_CACHE_LOADS) {
            self.record_query(
                query_name,
                self.incremental_load_result_event_kind,
                TimestampKind::End
            );
        }
    }

    #[inline]
    pub fn query_blocked_start(&self, query_name: QueryName) {
        if self.event_filter_mask.contains(EventFilter::QUERY_BLOCKED) {
            self.record_query(query_name, self.query_blocked_event_kind, TimestampKind::Start);
        }
    }

    #[inline]
    pub fn query_blocked_end(&self, query_name: QueryName) {
        if self.event_filter_mask.contains(EventFilter::QUERY_BLOCKED) {
            self.record_query(query_name, self.query_blocked_event_kind, TimestampKind::End);
        }
    }

    #[inline]
    fn record(&self, event_id: &str, event_kind: StringId, timestamp_kind: TimestampKind) {
        let thread_id = thread_id_to_u64(std::thread::current().id());

        let event_id = self.profiler.alloc_string(event_id);
        self.profiler.record_event(event_kind, event_id, thread_id, timestamp_kind);
    }

    #[inline]
    fn record_query(
        &self,
        query_name: QueryName,
        event_kind: StringId,
        timestamp_kind: TimestampKind,
    ) {
        let dep_node_name = SelfProfiler::get_query_name_string_id(query_name);

        let thread_id = thread_id_to_u64(std::thread::current().id());

        self.profiler.record_event(event_kind, dep_node_name, thread_id, timestamp_kind);
    }
}
