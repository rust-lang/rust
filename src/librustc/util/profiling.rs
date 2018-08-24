// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use session::config::Options;

use std::fs;
use std::io::{self, StdoutLock, Write};
use std::time::Instant;

macro_rules! define_categories {
    ($($name:ident,)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum ProfileCategory {
            $($name),*
        }

        #[allow(bad_style)]
        struct Categories<T> {
            $($name: T),*
        }

        impl<T: Default> Categories<T> {
            fn new() -> Categories<T> {
                Categories {
                    $($name: T::default()),*
                }
            }
        }

        impl<T> Categories<T> {
            fn get(&self, category: ProfileCategory) -> &T {
                match category {
                    $(ProfileCategory::$name => &self.$name),*
                }
            }

            fn set(&mut self, category: ProfileCategory, value: T) {
                match category {
                    $(ProfileCategory::$name => self.$name = value),*
                }
            }
        }

        struct CategoryData {
            times: Categories<u64>,
            query_counts: Categories<(u64, u64)>,
        }

        impl CategoryData {
            fn new() -> CategoryData {
                CategoryData {
                    times: Categories::new(),
                    query_counts: Categories::new(),
                }
            }

            fn print(&self, lock: &mut StdoutLock) {
                writeln!(lock, "| Phase            | Time (ms)      | Queries        | Hits (%) |")
                    .unwrap();
                writeln!(lock, "| ---------------- | -------------- | -------------- | -------- |")
                    .unwrap();

                $(
                    let (hits, total) = self.query_counts.$name;
                    let (hits, total) = if total > 0 {
                        (format!("{:.2}",
                        (((hits as f32) / (total as f32)) * 100.0)), total.to_string())
                    } else {
                        (String::new(), String::new())
                    };

                    writeln!(
                        lock,
                        "| {0: <16} | {1: <14} | {2: <14} | {3: <8} |",
                        stringify!($name),
                        self.times.$name / 1_000_000,
                        total,
                        hits
                    ).unwrap();
                )*
            }

            fn json(&self) -> String {
                let mut json = String::from("[");

                $(
                    let (hits, total) = self.query_counts.$name;

                    json.push_str(&format!(
                        "{{ \"category\": {}, \"time_ms\": {},
                            \"query_count\": {}, \"query_hits\": {} }}",
                        stringify!($name),
                        self.times.$name / 1_000_000,
                        total,
                        format!("{:.2}", (((hits as f32) / (total as f32)) * 100.0))
                    ));
                )*

                json.push(']');

                json
            }
        }
    }
}

define_categories! {
    Parsing,
    Expansion,
    TypeChecking,
    BorrowChecking,
    Codegen,
    Linking,
    Other,
}

pub struct SelfProfiler {
    timer_stack: Vec<ProfileCategory>,
    data: CategoryData,
    current_timer: Instant,
}

impl SelfProfiler {
    pub fn new() -> SelfProfiler {
        let mut profiler = SelfProfiler {
            timer_stack: Vec::new(),
            data: CategoryData::new(),
            current_timer: Instant::now(),
        };

        profiler.start_activity(ProfileCategory::Other);

        profiler
    }

    pub fn start_activity(&mut self, category: ProfileCategory) {
        match self.timer_stack.last().cloned() {
            None => {
                self.current_timer = Instant::now();
            },
            Some(current_category) if current_category == category => {
                //since the current category is the same as the new activity's category,
                //we don't need to do anything with the timer, we just need to push it on the stack
            }
            Some(current_category) => {
                let elapsed = self.stop_timer();

                //record the current category's time
                let new_time = self.data.times.get(current_category) + elapsed;
                self.data.times.set(current_category, new_time);
            }
        }

        //push the new category
        self.timer_stack.push(category);
    }

    pub fn record_query(&mut self, category: ProfileCategory) {
        let (hits, total) = *self.data.query_counts.get(category);
        self.data.query_counts.set(category, (hits, total + 1));
    }

    pub fn record_query_hit(&mut self, category: ProfileCategory) {
        let (hits, total) = *self.data.query_counts.get(category);
        self.data.query_counts.set(category, (hits + 1, total));
    }

    pub fn end_activity(&mut self, category: ProfileCategory) {
        match self.timer_stack.pop() {
            None => bug!("end_activity() was called but there was no running activity"),
            Some(c) =>
                assert!(
                    c == category,
                    "end_activity() was called but a different activity was running"),
        }

        //check if the new running timer is in the same category as this one
        //if it is, we don't need to do anything
        if let Some(c) = self.timer_stack.last() {
            if *c == category {
                return;
            }
        }

        //the new timer is different than the previous,
        //so record the elapsed time and start a new timer
        let elapsed = self.stop_timer();
        let new_time = self.data.times.get(category) + elapsed;
        self.data.times.set(category, new_time);
    }

    fn stop_timer(&mut self) -> u64 {
        let elapsed = self.current_timer.elapsed();

        self.current_timer = Instant::now();

        (elapsed.as_secs() * 1_000_000_000) + (elapsed.subsec_nanos() as u64)
    }

    pub fn print_results(&mut self, opts: &Options) {
        self.end_activity(ProfileCategory::Other);

        assert!(
            self.timer_stack.is_empty(),
            "there were timers running when print_results() was called");

        let out = io::stdout();
        let mut lock = out.lock();

        let crate_name =
            opts.crate_name
            .as_ref()
            .map(|n| format!(" for {}", n))
            .unwrap_or_default();

        writeln!(lock, "Self profiling results{}:", crate_name).unwrap();
        writeln!(lock).unwrap();

        self.data.print(&mut lock);

        writeln!(lock).unwrap();
        writeln!(lock, "Optimization level: {:?}", opts.optimize).unwrap();

        let incremental = if opts.incremental.is_some() { "on" } else { "off" };
        writeln!(lock, "Incremental: {}", incremental).unwrap();
    }

    pub fn save_results(&self, opts: &Options) {
        let category_data = self.data.json();
        let compilation_options =
            format!("{{ \"optimization_level\": \"{:?}\", \"incremental\": {} }}",
                    opts.optimize,
                    if opts.incremental.is_some() { "true" } else { "false" });

        let json = format!("{{ \"category_data\": {}, \"compilation_options\": {} }}",
                        category_data,
                        compilation_options);

        fs::write("self_profiler_results.json", json).unwrap();
    }
}
