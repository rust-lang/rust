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

use std::io::{self, StdoutLock, Write};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProfileCategory {
    Parsing,
    Expansion,
    TypeChecking,
    BorrowChecking,
    Codegen,
    Linking,
    Other,
}

struct Categories<T> {
    parsing: T,
    expansion: T,
    type_checking: T,
    borrow_checking: T,
    codegen: T,
    linking: T,
    other: T,
}

impl<T: Default> Categories<T> {
    fn new() -> Categories<T> {
        Categories {
            parsing: T::default(),
            expansion: T::default(),
            type_checking: T::default(),
            borrow_checking: T::default(),
            codegen: T::default(),
            linking: T::default(),
            other: T::default(),
        }
    }
}

impl<T> Categories<T> {
    fn get(&self, category: ProfileCategory) -> &T {
        match category {
            ProfileCategory::Parsing => &self.parsing,
            ProfileCategory::Expansion => &self.expansion,
            ProfileCategory::TypeChecking => &self.type_checking,
            ProfileCategory::BorrowChecking => &self.borrow_checking,
            ProfileCategory::Codegen => &self.codegen,
            ProfileCategory::Linking => &self.linking,
            ProfileCategory::Other => &self.other,
        }
    }

    fn set(&mut self, category: ProfileCategory, value: T) {
        match category {
            ProfileCategory::Parsing => self.parsing = value,
            ProfileCategory::Expansion => self.expansion = value,
            ProfileCategory::TypeChecking => self.type_checking = value,
            ProfileCategory::BorrowChecking => self.borrow_checking = value,
            ProfileCategory::Codegen => self.codegen = value,
            ProfileCategory::Linking => self.linking = value,
            ProfileCategory::Other => self.other = value,
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
        macro_rules! p {
            ($name:tt, $rustic_name:ident) => {
                writeln!(
                   lock,
                   "{0: <15} \t\t {1: <15}ms",
                   $name,
                   self.times.$rustic_name / 1_000_000
                ).unwrap();
                
                let (hits, total) = self.query_counts.$rustic_name;
                if total > 0 {
                    writeln!(
                        lock,
                        "\t{} hits {} queries",
                        hits,
                        total
                    ).unwrap();
                }
            };
        }

        p!("Parsing", parsing);
        p!("Expansion", expansion);
        p!("TypeChecking", type_checking);
        p!("BorrowChecking", borrow_checking);
        p!("Codegen", codegen);
        p!("Linking", linking);
        p!("Other", other);
    }
}

pub struct SelfProfiler {
    timer_stack: Vec<ProfileCategory>,
    data: CategoryData,
    current_timer: Instant,
}

pub struct ProfilerActivity<'a>(ProfileCategory, &'a mut SelfProfiler);

impl<'a> Drop for ProfilerActivity<'a> {
    fn drop(&mut self) {
        let ProfilerActivity (category, profiler) = self;

        profiler.end_activity(*category);
    }
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

        //the new timer is different than the previous, so record the elapsed time and start a new timer
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

        assert!(self.timer_stack.is_empty(), "there were timers running when print_results() was called");

        let out = io::stdout();
        let mut lock = out.lock();

        let crate_name = opts.crate_name.as_ref().map(|n| format!(" for {}", n)).unwrap_or_default();

        writeln!(lock, "Self profiling results{}:", crate_name).unwrap();

        self.data.print(&mut lock);

        writeln!(lock).unwrap();
        writeln!(lock, "Optimization level: {:?}", opts.optimize).unwrap();

        let incremental = if opts.incremental.is_some() { "on" } else { "off" };
        writeln!(lock, "Incremental: {}", incremental).unwrap();
    }

    pub fn record_activity<'a>(&'a mut self, category: ProfileCategory) -> ProfilerActivity<'a> {
        self.start_activity(category);

        ProfilerActivity(category, self)
    }
}