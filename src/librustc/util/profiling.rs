use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{self, Write};
use std::thread::ThreadId;
use std::time::Instant;

use crate::session::config::{Options, OptLevel};

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
    QueryCacheHit { query_name: &'static str, category: ProfileCategory },
    QueryCount { query_name: &'static str, category: ProfileCategory, count: usize },
    IncrementalLoadResultStart { query_name: &'static str, time: Instant },
    IncrementalLoadResultEnd { query_name: &'static str, time: Instant },
    QueryBlockedStart { query_name: &'static str, category: ProfileCategory, time: Instant },
    QueryBlockedEnd { query_name: &'static str, category: ProfileCategory, time: Instant },
}

impl ProfilerEvent {
    fn is_start_event(&self) -> bool {
        use self::ProfilerEvent::*;

        match self {
            QueryStart { .. } |
            GenericActivityStart { .. } |
            IncrementalLoadResultStart { .. } |
            QueryBlockedStart { .. } => true,

            QueryEnd { .. } |
            GenericActivityEnd { .. } |
            QueryCacheHit { .. } |
            QueryCount { .. } |
            IncrementalLoadResultEnd { .. } |
            QueryBlockedEnd { .. } => false,
        }
    }
}

pub struct SelfProfiler {
    events: HashMap<ThreadId, Vec<ProfilerEvent>>,
}

struct CategoryResultData {
    query_times: BTreeMap<&'static str, u64>,
    query_cache_stats: BTreeMap<&'static str, (u64, u64)>, //(hits, total)
}

impl CategoryResultData {
    fn new() -> CategoryResultData {
        CategoryResultData {
            query_times: BTreeMap::new(),
            query_cache_stats: BTreeMap::new(),
        }
    }

    fn total_time(&self) -> u64 {
        self.query_times.iter().map(|(_, time)| time).sum()
    }

    fn total_cache_data(&self) -> (u64, u64) {
        let (mut hits, mut total) = (0, 0);

        for (_, (h, t)) in &self.query_cache_stats {
            hits += h;
            total += t;
        }

        (hits, total)
    }
}

impl Default for CategoryResultData {
    fn default() -> CategoryResultData {
        CategoryResultData::new()
    }
}

struct CalculatedResults {
    categories: BTreeMap<ProfileCategory, CategoryResultData>,
    crate_name: Option<String>,
    optimization_level: OptLevel,
    incremental: bool,
    verbose: bool,
}

impl CalculatedResults {
    fn new() -> CalculatedResults {
        CalculatedResults {
            categories: BTreeMap::new(),
            crate_name: None,
            optimization_level: OptLevel::No,
            incremental: false,
            verbose: false,
        }
    }

    fn consolidate(mut cr1: CalculatedResults, cr2: CalculatedResults) -> CalculatedResults {
        for (category, data) in cr2.categories {
            let cr1_data = cr1.categories.entry(category).or_default();

            for (query, time) in data.query_times {
                *cr1_data.query_times.entry(query).or_default() += time;
            }

            for (query, (hits, total)) in data.query_cache_stats {
                let (h, t) = cr1_data.query_cache_stats.entry(query).or_insert((0, 0));
                *h += hits;
                *t += total;
            }
        }

        cr1
    }

    fn total_time(&self) -> u64 {
        self.categories.iter().map(|(_, data)| data.total_time()).sum()
    }

    fn with_options(mut self, opts: &Options) -> CalculatedResults {
        self.crate_name = opts.crate_name.clone();
        self.optimization_level = opts.optimize;
        self.incremental = opts.incremental.is_some();
        self.verbose = opts.debugging_opts.verbose;

        self
    }
}

fn time_between_ns(start: Instant, end: Instant) -> u64 {
    if start < end {
        let time = end - start;
        (time.as_secs() * 1_000_000_000) + (time.subsec_nanos() as u64)
    } else {
        debug!("time_between_ns: ignorning instance of end < start");
        0
    }
}

fn calculate_percent(numerator: u64, denominator: u64) -> f32 {
    if denominator > 0 {
        ((numerator as f32) / (denominator as f32)) * 100.0
    } else {
        0.0
    }
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
        })
    }

    #[inline]
    pub fn record_query_hit(&mut self, query_name: &'static str, category: ProfileCategory) {
        self.record(ProfilerEvent::QueryCacheHit {
            query_name,
            category,
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

    fn calculate_thread_results(events: &Vec<ProfilerEvent>) -> CalculatedResults {
        use self::ProfilerEvent::*;

        assert!(
            events.last().map(|e| !e.is_start_event()).unwrap_or(true),
            "there was an event running when calculate_reslts() was called"
        );

        let mut results = CalculatedResults::new();

        //(event, child time to subtract)
        let mut query_stack = Vec::new();

        for event in events {
            match event {
                QueryStart { .. } | GenericActivityStart { .. } => {
                    query_stack.push((event, 0));
                },
                QueryEnd { query_name, category, time: end_time } => {
                    let previous_query = query_stack.pop();
                    if let Some((QueryStart {
                                    query_name: p_query_name,
                                    time: start_time,
                                    category: _ }, child_time_to_subtract)) = previous_query {
                        assert_eq!(
                            p_query_name,
                            query_name,
                            "Saw a query end but the previous query wasn't the corresponding start"
                        );

                        let time_ns = time_between_ns(*start_time, *end_time);
                        let self_time_ns = time_ns - child_time_to_subtract;
                        let result_data = results.categories.entry(*category).or_default();

                        *result_data.query_times.entry(query_name).or_default() += self_time_ns;

                        if let Some((_, child_time_to_subtract)) = query_stack.last_mut() {
                            *child_time_to_subtract += time_ns;
                        }
                    } else {
                        bug!("Saw a query end but the previous event wasn't a query start");
                    }
                }
                GenericActivityEnd { category, time: end_time } => {
                    let previous_event = query_stack.pop();
                    if let Some((GenericActivityStart {
                                    category: previous_category,
                                    time: start_time }, child_time_to_subtract)) = previous_event {
                        assert_eq!(
                            previous_category,
                            category,
                            "Saw an end but the previous event wasn't the corresponding start"
                        );

                        let time_ns = time_between_ns(*start_time, *end_time);
                        let self_time_ns = time_ns - child_time_to_subtract;
                        let result_data = results.categories.entry(*category).or_default();

                        *result_data.query_times
                            .entry("{time spent not running queries}")
                            .or_default() += self_time_ns;

                        if let Some((_, child_time_to_subtract)) = query_stack.last_mut() {
                            *child_time_to_subtract += time_ns;
                        }
                    } else {
                        bug!("Saw an activity end but the previous event wasn't an activity start");
                    }
                },
                QueryCacheHit { category, query_name } => {
                    let result_data = results.categories.entry(*category).or_default();

                    let (hits, total) =
                        result_data.query_cache_stats.entry(query_name).or_insert((0, 0));
                    *hits += 1;
                    *total += 1;
                },
                QueryCount { category, query_name, count } => {
                    let result_data = results.categories.entry(*category).or_default();

                    let (_, totals) =
                        result_data.query_cache_stats.entry(query_name).or_insert((0, 0));
                    *totals += *count as u64;
                },
                //we don't summarize incremental load result events in the simple output mode
                IncrementalLoadResultStart { .. } | IncrementalLoadResultEnd { .. } => { },
                //we don't summarize parallel query blocking in the simple output mode
                QueryBlockedStart { .. } | QueryBlockedEnd { .. } => { },
            }
        }

        //normalize the times to ms
        for (_, data) in &mut results.categories {
            for (_, time) in &mut data.query_times {
                *time = *time / 1_000_000;
            }
        }

        results
    }

    fn get_results(&self, opts: &Options) -> CalculatedResults {
        self.events
            .iter()
            .map(|(_, r)| SelfProfiler::calculate_thread_results(r))
            .fold(CalculatedResults::new(), CalculatedResults::consolidate)
            .with_options(opts)
    }

    pub fn print_results(&mut self, opts: &Options) {
        self.end_activity(ProfileCategory::Other);

        let results = self.get_results(opts);

        let total_time = results.total_time() as f32;

        let out = io::stderr();
        let mut lock = out.lock();

        let crate_name = results.crate_name.map(|n| format!(" for {}", n)).unwrap_or_default();

        writeln!(lock, "Self profiling results{}:", crate_name).unwrap();
        writeln!(lock).unwrap();

        writeln!(lock, "| Phase                                     | Time (ms)      \
                        | Time (%) | Queries        | Hits (%)")
            .unwrap();
        writeln!(lock, "| ----------------------------------------- | -------------- \
                        | -------- | -------------- | --------")
            .unwrap();

        let mut categories: Vec<_> = results.categories.iter().collect();
        categories.sort_by_cached_key(|(_, d)| d.total_time());

        for (category, data) in categories.iter().rev() {
            let (category_hits, category_total) = data.total_cache_data();
            let category_hit_percent = calculate_percent(category_hits, category_total);

            writeln!(
                lock,
                "| {0: <41} | {1: >14} | {2: >8.2} | {3: >14} | {4: >8}",
                format!("{:?}", category),
                data.total_time(),
                ((data.total_time() as f32) / total_time) * 100.0,
                category_total,
                format!("{:.2}", category_hit_percent),
            ).unwrap();

            //in verbose mode, show individual query data
            if results.verbose {
                //don't show queries that took less than 1ms
                let mut times: Vec<_> = data.query_times.iter().filter(|(_, t)| **t > 0).collect();
                times.sort_by(|(_, time1), (_, time2)| time2.cmp(time1));

                for (query, time) in times {
                    let (hits, total) = data.query_cache_stats.get(query).unwrap_or(&(0, 0));
                    let hit_percent = calculate_percent(*hits, *total);

                    writeln!(
                        lock,
                        "| - {0: <39} | {1: >14} | {2: >8.2} | {3: >14} | {4: >8}",
                        query,
                        time,
                        ((*time as f32) / total_time) * 100.0,
                        total,
                        format!("{:.2}", hit_percent),
                    ).unwrap();
                }
            }
        }

        writeln!(lock).unwrap();
        writeln!(lock, "Optimization level: {:?}", opts.optimize).unwrap();
        writeln!(lock, "Incremental: {}", if results.incremental { "on" } else { "off" }).unwrap();
    }

    pub fn save_results(&self, opts: &Options) {
        let results = self.get_results(opts);

        let compilation_options =
            format!("{{ \"optimization_level\": \"{:?}\", \"incremental\": {} }}",
                    results.optimization_level,
                    if results.incremental { "true" } else { "false" });

        let mut category_data = String::new();

        for (category, data) in &results.categories {
            let (hits, total) = data.total_cache_data();
            let hit_percent = calculate_percent(hits, total);

            category_data.push_str(&format!("{{ \"category\": \"{:?}\", \"time_ms\": {}, \
                                                \"query_count\": {}, \"query_hits\": {} }}",
                                            category,
                                            data.total_time(),
                                            total,
                                            format!("{:.2}", hit_percent)));
        }

        let json = format!("{{ \"category_data\": {}, \"compilation_options\": {} }}",
                        category_data,
                        compilation_options);

        fs::write("self_profiler_results.json", json).unwrap();
    }
}
