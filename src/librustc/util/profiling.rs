use std::borrow::Cow;
use std::error::Error;
use std::mem::{self, Discriminant};
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProfilerEvent {
    QueryStart { query_name: &'static str, category: ProfileCategory, time: u64 },
    QueryEnd { query_name: &'static str, category: ProfileCategory, time: u64 },
    GenericActivityStart { category: ProfileCategory, label: Cow<'static, str>, time: u64 },
    GenericActivityEnd { category: ProfileCategory, label: Cow<'static, str>, time: u64 },
    IncrementalLoadResultStart { query_name: &'static str, time: u64 },
    IncrementalLoadResultEnd { query_name: &'static str, time: u64 },
    QueryCacheHit { query_name: &'static str, category: ProfileCategory, time: u64 },
    QueryCount { query_name: &'static str, category: ProfileCategory, count: usize, time: u64 },
    QueryBlockedStart { query_name: &'static str, category: ProfileCategory, time: u64 },
    QueryBlockedEnd { query_name: &'static str, category: ProfileCategory, time: u64 },
}

fn thread_id_to_u64(tid: ThreadId) -> u64 {
    unsafe { mem::transmute::<ThreadId, u64>(tid) }
}

pub struct SelfProfiler {
    profiler: Profiler,
    query_event_kind: StringId,
    generic_activity_event_kind: StringId,
    incremental_load_result_event_kind: StringId,
    query_blocked_event_kind: StringId,
    query_cache_hit_event_kind: StringId,
}

impl SelfProfiler {
    pub fn new() -> Result<SelfProfiler, Box<dyn Error>> {
        let filename = format!("pid-{}.rustc_profile", process::id());
        let path = std::path::Path::new(&filename);
        let profiler = Profiler::new(path)?;

        let query_event_kind = profiler.alloc_string("Query");
        let generic_activity_event_kind = profiler.alloc_string("GenericActivity");
        let incremental_load_result_event_kind = profiler.alloc_string("IncrementalLoadResult");
        let query_blocked_event_kind = profiler.alloc_string("QueryBlocked");
        let query_cache_hit_event_kind = profiler.alloc_string("QueryCacheHit");

        Ok(SelfProfiler {
            profiler,
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
        self.record(&label.into(), self.generic_activity_event_kind, TimestampKind::Start);
    }

    #[inline]
    pub fn end_activity(
        &self,
        label: impl Into<Cow<'static, str>>,
    ) {
        self.record(&label.into(), self.generic_activity_event_kind, TimestampKind::End);
    }

    #[inline]
    pub fn record_query_hit(&self, query_name: QueryName) {
        self.record_query(query_name, self.query_cache_hit_event_kind, TimestampKind::Instant);
    }

    #[inline]
    pub fn start_query(&self, query_name: QueryName) {
        self.record_query(query_name, self.query_event_kind, TimestampKind::Start);
    }

    #[inline]
    pub fn end_query(&self, query_name: QueryName) {
        self.record_query(query_name, self.query_event_kind, TimestampKind::End);
    }

    #[inline]
    pub fn incremental_load_result_start(&self, query_name: QueryName) {
        self.record_query(
            query_name,
            self.incremental_load_result_event_kind,
            TimestampKind::Start
        );
    }

    #[inline]
    pub fn incremental_load_result_end(&self, query_name: QueryName) {
        self.record_query(query_name, self.incremental_load_result_event_kind, TimestampKind::End);
    }

    #[inline]
    pub fn query_blocked_start(&self, query_name: QueryName) {
        self.record_query(query_name, self.query_blocked_event_kind, TimestampKind::Start);
    }

    #[inline]
    pub fn query_blocked_end(&self, query_name: QueryName) {
        self.record_query(query_name, self.query_blocked_event_kind, TimestampKind::End);
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
