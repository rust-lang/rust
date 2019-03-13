use std::fs;
use std::io::{BufWriter, Write};
use std::mem;
use std::process;
use std::thread::ThreadId;
use std::time::{Duration, Instant, SystemTime};

use crate::session::config::Options;

use rustc_data_structures::fx::FxHashMap;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfilerEvent {
    QueryStart { query_name: &'static str, category: ProfileCategory, time: u64 },
    QueryEnd { query_name: &'static str, category: ProfileCategory, time: u64 },
    GenericActivityStart { category: ProfileCategory, time: u64 },
    GenericActivityEnd { category: ProfileCategory, time: u64 },
    IncrementalLoadResultStart { query_name: &'static str, time: u64 },
    IncrementalLoadResultEnd { query_name: &'static str, time: u64 },
    QueryCacheHit { query_name: &'static str, category: ProfileCategory, time: u64 },
    QueryCount { query_name: &'static str, category: ProfileCategory, count: usize, time: u64 },
    QueryBlockedStart { query_name: &'static str, category: ProfileCategory, time: u64 },
    QueryBlockedEnd { query_name: &'static str, category: ProfileCategory, time: u64 },
}

impl ProfilerEvent {
    fn timestamp(&self) -> u64 {
        use self::ProfilerEvent::*;

        match self {
            QueryStart { time, .. } |
            QueryEnd { time, .. } |
            GenericActivityStart { time, .. } |
            GenericActivityEnd { time, .. } |
            QueryCacheHit { time, .. } |
            QueryCount { time, .. } |
            IncrementalLoadResultStart { time, .. } |
            IncrementalLoadResultEnd { time, .. } |
            QueryBlockedStart { time, .. } |
            QueryBlockedEnd { time, .. } => *time
        }
    }
}

fn thread_id_to_u64(tid: ThreadId) -> u64 {
    unsafe { mem::transmute::<ThreadId, u64>(tid) }
}

pub struct SelfProfiler {
    events: FxHashMap<ThreadId, Vec<ProfilerEvent>>,
    start_time: SystemTime,
    start_instant: Instant,
}

impl SelfProfiler {
    pub fn new() -> SelfProfiler {
        let profiler = SelfProfiler {
            events: Default::default(),
            start_time: SystemTime::now(),
            start_instant: Instant::now(),
        };

        profiler
    }

    #[inline]
    pub fn start_activity(&mut self, category: ProfileCategory) {
        self.record(ProfilerEvent::GenericActivityStart {
            category,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn end_activity(&mut self, category: ProfileCategory) {
        self.record(ProfilerEvent::GenericActivityEnd {
            category,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn record_computed_queries(
        &mut self,
        query_name: &'static str,
        category: ProfileCategory,
        count: usize)
        {
        self.record(ProfilerEvent::QueryCount {
            query_name,
            category,
            count,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn record_query_hit(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryCacheHit {
            query_name,
            category,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn start_query(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryStart {
            query_name,
            category,
            time: self.get_time_from_start(),
        });
    }

    #[inline]
    pub fn end_query(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryEnd {
            query_name,
            category,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn incremental_load_result_start(&mut self, query_name: &'static str) {
        self.record(ProfilerEvent::IncrementalLoadResultStart {
            query_name,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn incremental_load_result_end(&mut self, query_name: &'static str) {
        self.record(ProfilerEvent::IncrementalLoadResultEnd {
            query_name,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn query_blocked_start(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryBlockedStart {
            query_name,
            category,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    pub fn query_blocked_end(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryBlockedEnd {
            query_name,
            category,
            time: self.get_time_from_start(),
        })
    }

    #[inline]
    fn record(&mut self, event: ProfilerEvent) {
        let thread_id = std::thread::current().id();
        let events = self.events.entry(thread_id).or_default();

        events.push(event);
    }

    #[inline]
    fn get_time_from_start(&self) -> u64 {
        let duration = Instant::now() - self.start_instant;
        duration.as_nanos() as u64
    }

    pub fn dump_raw_events(&self, opts: &Options) {
        use self::ProfilerEvent::*;

        let pid = process::id();

        let filename =
            format!("{}.profile_events.json", opts.crate_name.clone().unwrap_or_default());

        let mut file = BufWriter::new(fs::File::create(filename).unwrap());

        let threads: Vec<_> =
            self.events
                .keys()
                .into_iter()
                .map(|tid| format!("{}", thread_id_to_u64(*tid)))
                .collect();

        write!(file,
            "{{\
                \"processes\": {{\
                    \"{}\": {{\
                        \"threads\": [{}],\
                        \"crate_name\": \"{}\",\
                        \"opt_level\": \"{:?}\",\
                        \"incremental\": {}\
                    }}\
                }},\
                \"events\": [\
             ",
            pid,
            threads.join(","),
            opts.crate_name.clone().unwrap_or_default(),
            opts.optimize,
            if opts.incremental.is_some() { "true" } else { "false" },
        ).unwrap();

        let mut is_first = true;
        for (thread_id, events) in &self.events {
            let thread_id = thread_id_to_u64(*thread_id);

            for event in events {
                if is_first {
                    is_first = false;
                } else {
                    writeln!(file, ",").unwrap();
                }

                let (secs, nanos) = {
                    let time = self.start_time + Duration::from_nanos(event.timestamp());
                    let time_since_unix =
                        time.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
                    (time_since_unix.as_secs(), time_since_unix.subsec_nanos())
                };

                match event {
                    QueryStart { query_name, category, time: _ } =>
                        write!(file,
                            "{{ \
                                \"QueryStart\": {{ \
                                    \"query_name\": \"{}\",\
                                    \"category\": \"{:?}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            category,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    QueryEnd { query_name, category, time: _ } =>
                        write!(file,
                            "{{\
                                \"QueryEnd\": {{\
                                    \"query_name\": \"{}\",\
                                    \"category\": \"{:?}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            category,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    GenericActivityStart { category, time: _ } =>
                        write!(file,
                            "{{
                                \"GenericActivityStart\": {{\
                                    \"category\": \"{:?}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            category,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    GenericActivityEnd { category, time: _ } =>
                        write!(file,
                            "{{\
                                \"GenericActivityEnd\": {{\
                                    \"category\": \"{:?}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            category,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    QueryCacheHit { query_name, category, time: _ } =>
                        write!(file,
                            "{{\
                                \"QueryCacheHit\": {{\
                                    \"query_name\": \"{}\",\
                                    \"category\": \"{:?}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            category,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    QueryCount { query_name, category, count, time: _ } =>
                        write!(file,
                            "{{\
                                \"QueryCount\": {{\
                                    \"query_name\": \"{}\",\
                                    \"category\": \"{:?}\",\
                                    \"count\": {},\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            category,
                            count,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    IncrementalLoadResultStart { query_name, time: _ } =>
                        write!(file,
                            "{{\
                                \"IncrementalLoadResultStart\": {{\
                                    \"query_name\": \"{}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    IncrementalLoadResultEnd { query_name, time: _ } =>
                        write!(file,
                            "{{\
                                \"IncrementalLoadResultEnd\": {{\
                                    \"query_name\": \"{}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    QueryBlockedStart { query_name, category, time: _ } =>
                        write!(file,
                            "{{\
                                \"QueryBlockedStart\": {{\
                                    \"query_name\": \"{}\",\
                                    \"category\": \"{:?}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            category,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap(),
                    QueryBlockedEnd { query_name, category, time: _ } =>
                        write!(file,
                            "{{\
                                \"QueryBlockedEnd\": {{\
                                    \"query_name\": \"{}\",\
                                    \"category\": \"{:?}\",\
                                    \"time\": {{\
                                        \"secs\": {},\
                                        \"nanos\": {}\
                                    }},\
                                    \"thread_id\": {}\
                                }}\
                            }}",
                            query_name,
                            category,
                            secs,
                            nanos,
                            thread_id,
                        ).unwrap()
                }
            }
        }

        write!(file, "] }}").unwrap();
    }
}
