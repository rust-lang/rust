use rustc_data_structures::fx::FxHashMap;
use std::fs::File;
use std::io::prelude::*;
use std::marker::PhantomData;
use std::mem;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const OUTPUT_WIDTH_IN_PX: u64 = 1000;
const TIME_LINE_HEIGHT_IN_PX: u64 = 20;
const TIME_LINE_HEIGHT_STRIDE_IN_PX: usize = 30;

#[derive(Clone)]
struct Timing {
    start: Instant,
    end: Instant,
    work_package_kind: WorkPackageKind,
    name: String,
    events: Vec<(String, Instant)>,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct TimelineId(pub usize);

#[derive(Clone)]
struct PerThread {
    timings: Vec<Timing>,
    open_work_package: Option<(Instant, WorkPackageKind, String)>,
}

#[derive(Clone)]
pub struct TimeGraph {
    data: Arc<Mutex<FxHashMap<TimelineId, PerThread>>>,
}

#[derive(Clone, Copy)]
pub struct WorkPackageKind(pub &'static [&'static str]);

pub struct Timeline {
    token: Option<RaiiToken>,
}

struct RaiiToken {
    graph: TimeGraph,
    timeline: TimelineId,
    events: Vec<(String, Instant)>,
    // The token must not be Send:
    _marker: PhantomData<*const ()>
}


impl Drop for RaiiToken {
    fn drop(&mut self) {
        self.graph.end(self.timeline, mem::replace(&mut self.events, Vec::new()));
    }
}

impl TimeGraph {
    pub fn new() -> TimeGraph {
        TimeGraph {
            data: Arc::new(Mutex::new(FxHashMap::default()))
        }
    }

    pub fn start(&self,
                 timeline: TimelineId,
                 work_package_kind: WorkPackageKind,
                 name: &str) -> Timeline {
        {
            let mut table = self.data.lock().unwrap();

            let data = table.entry(timeline).or_insert(PerThread {
                timings: Vec::new(),
                open_work_package: None,
            });

            assert!(data.open_work_package.is_none());
            data.open_work_package = Some((Instant::now(), work_package_kind, name.to_string()));
        }

        Timeline {
            token: Some(RaiiToken {
                graph: self.clone(),
                timeline,
                events: Vec::new(),
                _marker: PhantomData,
            }),
        }
    }

    fn end(&self, timeline: TimelineId, events: Vec<(String, Instant)>) {
        let end = Instant::now();

        let mut table = self.data.lock().unwrap();
        let data = table.get_mut(&timeline).unwrap();

        if let Some((start, work_package_kind, name)) = data.open_work_package.take() {
            data.timings.push(Timing {
                start,
                end,
                work_package_kind,
                name,
                events,
            });
        } else {
            bug!("end timing without start?")
        }
    }

    pub fn dump(&self, output_filename: &str) {
        let table = self.data.lock().unwrap();

        for data in table.values() {
            assert!(data.open_work_package.is_none());
        }

        let mut threads: Vec<PerThread> =
            table.values().map(|data| data.clone()).collect();

        threads.sort_by_key(|timeline| timeline.timings[0].start);

        let earliest_instant = threads[0].timings[0].start;
        let latest_instant = threads.iter()
                                       .map(|timeline| timeline.timings
                                                               .last()
                                                               .unwrap()
                                                               .end)
                                       .max()
                                       .unwrap();
        let max_distance = distance(earliest_instant, latest_instant);

        let mut file = File::create(format!("{}.html", output_filename)).unwrap();

        writeln!(file, "
            <html>
            <head>
                <style>
                    #threads a {{
                        position: absolute;
                        overflow: hidden;
                    }}
                    #threads {{
                        height: {total_height}px;
                        width: {width}px;
                    }}

                    .timeline {{
                        display: none;
                        width: {width}px;
                        position: relative;
                    }}

                    .timeline:target {{
                        display: block;
                    }}

                    .event {{
                        position: absolute;
                    }}
                </style>
            </head>
            <body>
                <div id='threads'>
        ",
            total_height = threads.len() * TIME_LINE_HEIGHT_STRIDE_IN_PX,
            width = OUTPUT_WIDTH_IN_PX,
        ).unwrap();

        let mut color = 0;
        for (line_index, thread) in threads.iter().enumerate() {
            let line_top = line_index * TIME_LINE_HEIGHT_STRIDE_IN_PX;

            for span in &thread.timings {
                let start = distance(earliest_instant, span.start);
                let end = distance(earliest_instant, span.end);

                let start = normalize(start, max_distance, OUTPUT_WIDTH_IN_PX);
                let end = normalize(end, max_distance, OUTPUT_WIDTH_IN_PX);

                let colors = span.work_package_kind.0;

                writeln!(file, "<a href='#timing{}'
                                   style='top:{}px; \
                                          left:{}px; \
                                          width:{}px; \
                                          height:{}px; \
                                          background:{};'>{}</a>",
                    color,
                    line_top,
                    start,
                    end - start,
                    TIME_LINE_HEIGHT_IN_PX,
                    colors[color % colors.len()],
                    span.name,
                    ).unwrap();

                color += 1;
            }
        }

        writeln!(file, "
            </div>
        ").unwrap();

        let mut idx = 0;
        for thread in threads.iter() {
            for timing in &thread.timings {
                let colors = timing.work_package_kind.0;
                let height = TIME_LINE_HEIGHT_STRIDE_IN_PX * timing.events.len();
                writeln!(file, "<div class='timeline'
                                     id='timing{}'
                                     style='background:{};height:{}px;'>",
                         idx,
                         colors[idx % colors.len()],
                         height).unwrap();
                idx += 1;
                let max = distance(timing.start, timing.end);
                for (i, &(ref event, time)) in timing.events.iter().enumerate() {
                    let i = i as u64;
                    let time = distance(timing.start, time);
                    let at = normalize(time, max, OUTPUT_WIDTH_IN_PX);
                    writeln!(file, "<span class='event'
                                          style='left:{}px;\
                                                 top:{}px;'>{}</span>",
                             at,
                             TIME_LINE_HEIGHT_IN_PX * i,
                             event).unwrap();
                }
                writeln!(file, "</div>").unwrap();
            }
        }

        writeln!(file, "
            </body>
            </html>
        ").unwrap();
    }
}

impl Timeline {
    pub fn noop() -> Timeline {
        Timeline { token: None }
    }

    /// Record an event which happened at this moment on this timeline.
    ///
    /// Events are displayed in the eventual HTML output where you can click on
    /// a particular timeline and it'll expand to all of the events that
    /// happened on that timeline. This can then be used to drill into a
    /// particular timeline and see what events are happening and taking the
    /// most time.
    pub fn record(&mut self, name: &str) {
        if let Some(ref mut token) = self.token {
            token.events.push((name.to_string(), Instant::now()));
        }
    }
}

fn distance(zero: Instant, x: Instant) -> u64 {

    let duration = x.duration_since(zero);
    (duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64) // / div
}

fn normalize(distance: u64, max: u64, max_pixels: u64) -> u64 {
    (max_pixels * distance) / max
}

