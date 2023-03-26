//! This module is responsible for collecting metrics profiling information for the current build
//! and dumping it to disk as JSON, to aid investigations on build and CI performance.
//!
//! As this module requires additional dependencies not present during local builds, it's cfg'd
//! away whenever the `build.metrics` config option is not set to `true`.

use crate::builder::Step;
use crate::util::t;
use crate::Build;
use serde_derive::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs::File;
use std::io::BufWriter;
use std::time::{Duration, Instant};
use sysinfo::{CpuExt, System, SystemExt};

pub(crate) struct BuildMetrics {
    state: RefCell<MetricsState>,
}

impl BuildMetrics {
    pub(crate) fn init() -> Self {
        let state = RefCell::new(MetricsState {
            finished_steps: Vec::new(),
            running_steps: Vec::new(),

            system_info: System::new(),
            timer_start: None,
            invocation_timer_start: Instant::now(),
        });

        BuildMetrics { state }
    }

    pub(crate) fn enter_step<S: Step>(&self, step: &S) {
        let mut state = self.state.borrow_mut();

        // Consider all the stats gathered so far as the parent's.
        if !state.running_steps.is_empty() {
            self.collect_stats(&mut *state);
        }

        state.system_info.refresh_cpu();
        state.timer_start = Some(Instant::now());

        state.running_steps.push(StepMetrics {
            type_: std::any::type_name::<S>().into(),
            debug_repr: format!("{step:?}"),

            cpu_usage_time_sec: 0.0,
            duration_excluding_children_sec: Duration::ZERO,

            children: Vec::new(),
            tests: Vec::new(),
        });
    }

    pub(crate) fn exit_step(&self) {
        let mut state = self.state.borrow_mut();

        self.collect_stats(&mut *state);

        let step = state.running_steps.pop().unwrap();
        if state.running_steps.is_empty() {
            state.finished_steps.push(step);
            state.timer_start = None;
        } else {
            state.running_steps.last_mut().unwrap().children.push(step);

            // Start collecting again for the parent step.
            state.system_info.refresh_cpu();
            state.timer_start = Some(Instant::now());
        }
    }

    pub(crate) fn record_test(&self, name: &str, outcome: TestOutcome) {
        let mut state = self.state.borrow_mut();
        state
            .running_steps
            .last_mut()
            .unwrap()
            .tests
            .push(Test { name: name.to_string(), outcome });
    }

    fn collect_stats(&self, state: &mut MetricsState) {
        let step = state.running_steps.last_mut().unwrap();

        let elapsed = state.timer_start.unwrap().elapsed();
        step.duration_excluding_children_sec += elapsed;

        state.system_info.refresh_cpu();
        let cpu = state.system_info.cpus().iter().map(|p| p.cpu_usage()).sum::<f32>();
        step.cpu_usage_time_sec += cpu as f64 / 100.0 * elapsed.as_secs_f64();
    }

    pub(crate) fn persist(&self, build: &Build) {
        let mut state = self.state.borrow_mut();
        assert!(state.running_steps.is_empty(), "steps are still executing");

        let dest = build.out.join("metrics.json");

        let mut system = System::new();
        system.refresh_cpu();
        system.refresh_memory();

        let system_stats = JsonInvocationSystemStats {
            cpu_threads_count: system.cpus().len(),
            cpu_model: system.cpus()[0].brand().into(),

            memory_total_bytes: system.total_memory(),
        };
        let steps = std::mem::take(&mut state.finished_steps);

        // Some of our CI builds consist of multiple independent CI invocations. Ensure all the
        // previous invocations are still present in the resulting file.
        let mut invocations = match std::fs::read(&dest) {
            Ok(contents) => t!(serde_json::from_slice::<JsonRoot>(&contents)).invocations,
            Err(err) => {
                if err.kind() != std::io::ErrorKind::NotFound {
                    panic!("failed to open existing metrics file at {}: {err}", dest.display());
                }
                Vec::new()
            }
        };
        invocations.push(JsonInvocation {
            duration_including_children_sec: state.invocation_timer_start.elapsed().as_secs_f64(),
            children: steps.into_iter().map(|step| self.prepare_json_step(step)).collect(),
        });

        let json = JsonRoot { system_stats, invocations };

        t!(std::fs::create_dir_all(dest.parent().unwrap()));
        let mut file = BufWriter::new(t!(File::create(&dest)));
        t!(serde_json::to_writer(&mut file, &json));
    }

    fn prepare_json_step(&self, step: StepMetrics) -> JsonNode {
        let mut children = Vec::new();
        children.extend(step.children.into_iter().map(|child| self.prepare_json_step(child)));
        children.extend(
            step.tests
                .into_iter()
                .map(|test| JsonNode::Test { name: test.name, outcome: test.outcome }),
        );

        JsonNode::RustbuildStep {
            type_: step.type_,
            debug_repr: step.debug_repr,

            duration_excluding_children_sec: step.duration_excluding_children_sec.as_secs_f64(),
            system_stats: JsonStepSystemStats {
                cpu_utilization_percent: step.cpu_usage_time_sec * 100.0
                    / step.duration_excluding_children_sec.as_secs_f64(),
            },

            children,
        }
    }
}

struct MetricsState {
    finished_steps: Vec<StepMetrics>,
    running_steps: Vec<StepMetrics>,

    system_info: System,
    timer_start: Option<Instant>,
    invocation_timer_start: Instant,
}

struct StepMetrics {
    type_: String,
    debug_repr: String,

    cpu_usage_time_sec: f64,
    duration_excluding_children_sec: Duration,

    children: Vec<StepMetrics>,
    tests: Vec<Test>,
}

struct Test {
    name: String,
    outcome: TestOutcome,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct JsonRoot {
    system_stats: JsonInvocationSystemStats,
    invocations: Vec<JsonInvocation>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct JsonInvocation {
    duration_including_children_sec: f64,
    children: Vec<JsonNode>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum JsonNode {
    RustbuildStep {
        #[serde(rename = "type")]
        type_: String,
        debug_repr: String,

        duration_excluding_children_sec: f64,
        system_stats: JsonStepSystemStats,

        children: Vec<JsonNode>,
    },
    Test {
        name: String,
        #[serde(flatten)]
        outcome: TestOutcome,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub(crate) enum TestOutcome {
    Passed,
    Failed,
    Ignored { ignore_reason: Option<String> },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct JsonInvocationSystemStats {
    cpu_threads_count: usize,
    cpu_model: String,

    memory_total_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct JsonStepSystemStats {
    cpu_utilization_percent: f64,
}
