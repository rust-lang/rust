// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::io::prelude::*;
use std::fs::File;

const OUTPUT_WIDTH_IN_PX: u64 = 1000;
const TIME_LINE_HEIGHT_IN_PX: u64 = 7;
const TIME_LINE_HEIGHT_STRIDE_IN_PX: usize = 10;

#[derive(Clone)]
struct Timing {
    start: Instant,
    end: Instant,
    work_package_kind: WorkPackageKind,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct TimelineId(pub usize);

#[derive(Clone)]
struct PerThread {
    timings: Vec<Timing>,
    open_work_package: Option<(Instant, WorkPackageKind)>,
}

#[derive(Clone)]
pub struct TimeGraph {
    data: Arc<Mutex<HashMap<TimelineId, PerThread>>>,
}

#[derive(Clone, Copy)]
pub struct WorkPackageKind(pub &'static [&'static str]);

pub struct RaiiToken {
    graph: TimeGraph,
    timeline: TimelineId,
    // The token must not be Send:
    _marker: PhantomData<*const ()>
}


impl Drop for RaiiToken {
    fn drop(&mut self) {
        self.graph.end(self.timeline);
    }
}

impl TimeGraph {
    pub fn new() -> TimeGraph {
        TimeGraph {
            data: Arc::new(Mutex::new(HashMap::new()))
        }
    }

    pub fn start(&self,
                 timeline: TimelineId,
                 work_package_kind: WorkPackageKind) -> RaiiToken {
        {
            let mut table = self.data.lock().unwrap();

            let mut data = table.entry(timeline).or_insert(PerThread {
                timings: Vec::new(),
                open_work_package: None,
            });

            assert!(data.open_work_package.is_none());
            data.open_work_package = Some((Instant::now(), work_package_kind));
        }

        RaiiToken {
            graph: self.clone(),
            timeline,
            _marker: PhantomData,
        }
    }

    fn end(&self, timeline: TimelineId) {
        let end = Instant::now();

        let mut table = self.data.lock().unwrap();
        let mut data = table.get_mut(&timeline).unwrap();

        if let Some((start, work_package_kind)) = data.open_work_package {
            data.timings.push(Timing {
                start,
                end,
                work_package_kind,
            });
        } else {
            bug!("end timing without start?")
        }

        data.open_work_package = None;
    }

    pub fn dump(&self, output_filename: &str) {
        let table = self.data.lock().unwrap();

        for data in table.values() {
            assert!(data.open_work_package.is_none());
        }

        let mut timelines: Vec<PerThread> =
            table.values().map(|data| data.clone()).collect();

        timelines.sort_by_key(|timeline| timeline.timings[0].start);

        let earliest_instant = timelines[0].timings[0].start;
        let latest_instant = timelines.iter()
                                       .map(|timeline| timeline.timings
                                                               .last()
                                                               .unwrap()
                                                               .end)
                                       .max()
                                       .unwrap();
        let max_distance = distance(earliest_instant, latest_instant);

        let mut file = File::create(format!("{}.html", output_filename)).unwrap();

        writeln!(file, "<html>").unwrap();
        writeln!(file, "<head></head>").unwrap();
        writeln!(file, "<body>").unwrap();

        let mut color = 0;

        for (line_index, timeline) in timelines.iter().enumerate() {
            let line_top = line_index * TIME_LINE_HEIGHT_STRIDE_IN_PX;

            for span in &timeline.timings {
                let start = distance(earliest_instant, span.start);
                let end = distance(earliest_instant, span.end);

                let start = normalize(start, max_distance, OUTPUT_WIDTH_IN_PX);
                let end = normalize(end, max_distance, OUTPUT_WIDTH_IN_PX);

                let colors = span.work_package_kind.0;

                writeln!(file, "<div style='position:absolute; \
                                            top:{}px; \
                                            left:{}px; \
                                            width:{}px; \
                                            height:{}px; \
                                            background:{};'></div>",
                    line_top,
                    start,
                    end - start,
                    TIME_LINE_HEIGHT_IN_PX,
                    colors[color % colors.len()]
                    ).unwrap();

                color += 1;
            }
        }

        writeln!(file, "</body>").unwrap();
        writeln!(file, "</html>").unwrap();
    }
}

fn distance(zero: Instant, x: Instant) -> u64 {

    let duration = x.duration_since(zero);
    (duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64) // / div
}

fn normalize(distance: u64, max: u64, max_pixels: u64) -> u64 {
    (max_pixels * distance) / max
}

