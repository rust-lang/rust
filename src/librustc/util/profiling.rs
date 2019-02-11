use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::mem;
use std::process;
use std::thread::ThreadId;
use std::time::Instant;

use crate::session::config::Options;

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
    QueryStart { query_name: &'static str, category: ProfileCategory, time: Instant },
    QueryEnd { query_name: &'static str, category: ProfileCategory, time: Instant },
    GenericActivityStart { category: ProfileCategory, time: Instant },
    GenericActivityEnd { category: ProfileCategory, time: Instant },
    IncrementalLoadResultStart { query_name: &'static str, time: Instant },
    IncrementalLoadResultEnd { query_name: &'static str, time: Instant },
    QueryCacheHit { query_name: &'static str, category: ProfileCategory, time: Instant },
    QueryCount { query_name: &'static str, category: ProfileCategory, count: usize, time: Instant },
    QueryBlockedStart { query_name: &'static str, category: ProfileCategory, time: Instant },
    QueryBlockedEnd { query_name: &'static str, category: ProfileCategory, time: Instant },
}

impl ProfilerEvent {
    fn timestamp(&self) -> Instant {
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
    events: HashMap<ThreadId, Vec<ProfilerEvent>>,
}

impl SelfProfiler {
    pub fn new() -> SelfProfiler {
        let mut profiler = SelfProfiler {
            events: HashMap::new(),
        };

        profiler.start_activity(ProfileCategory::Other);

        profiler
    }

    #[inline]
    pub fn start_activity(&mut self, category: ProfileCategory) {
        self.record(ProfilerEvent::GenericActivityStart {
            category,
            time: Instant::now(),
        })
    }

    #[inline]
    pub fn end_activity(&mut self, category: ProfileCategory) {
        self.record(ProfilerEvent::GenericActivityEnd {
            category,
            time: Instant::now(),
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
            time: Instant::now(),
        })
    }

    #[inline]
    pub fn record_query_hit(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryCacheHit {
            query_name,
            category,
            time: Instant::now(),
        })
    }

    #[inline]
    pub fn start_query(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryStart {
            query_name,
            category,
            time: Instant::now(),
        });
    }

    #[inline]
    pub fn end_query(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryEnd {
            query_name,
            category,
            time: Instant::now(),
        })
    }

    #[inline]
    pub fn incremental_load_result_start(&mut self, query_name: &'static str) {
        self.record(ProfilerEvent::IncrementalLoadResultStart {
            query_name,
            time: Instant::now(),
        })
    }

    #[inline]
    pub fn incremental_load_result_end(&mut self, query_name: &'static str) {
        self.record(ProfilerEvent::IncrementalLoadResultEnd {
            query_name,
            time: Instant::now(),
        })
    }

    #[inline]
    pub fn query_blocked_start(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryBlockedStart {
            query_name,
            category,
            time: Instant::now(),
        })
    }

    #[inline]
    pub fn query_blocked_end(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryBlockedEnd {
            query_name,
            category,
            time: Instant::now(),
        })
    }

    #[inline]
    fn record(&mut self, event: ProfilerEvent) {
        let thread_id = std::thread::current().id();
        let events = self.events.entry(thread_id).or_default();

        events.push(event);
    }

    pub fn dump_raw_events(&self, opts: &Options) {
        use self::ProfilerEvent::*;

        //find the earliest Instant to use as t=0
        //when serializing the events, we'll calculate a Duration
        //using (instant - min_instant)
        let min_instant =
            self.events
                .iter()
                .map(|(_, values)| values[0].timestamp())
                .min()
                .unwrap();

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
                    let duration = event.timestamp() - min_instant;
                    (duration.as_secs(), duration.subsec_nanos())
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
