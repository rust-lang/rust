//! Collects statistics and emits suite/test events as JSON messages, using
//! the same JSON format as libtest's JSON formatter.
//!
//! These messages are then parsed by bootstrap, which replaces them with
//! user-friendly terminal output.

use std::time::Instant;

use serde_json::json;

use crate::executor::{CollectedTest, TestCompletion, TestOutcome};

pub(crate) struct Listener {
    suite_start: Option<Instant>,
    passed: usize,
    failed: usize,
    ignored: usize,
    filtered_out: usize,
}

impl Listener {
    pub(crate) fn new() -> Self {
        Self { suite_start: None, passed: 0, failed: 0, ignored: 0, filtered_out: 0 }
    }

    fn print_message(&self, message: &serde_json::Value) {
        println!("{message}");
    }

    fn now(&self) -> Instant {
        Instant::now()
    }

    pub(crate) fn suite_started(&mut self, test_count: usize, filtered_out: usize) {
        self.suite_start = Some(self.now());
        self.filtered_out = filtered_out;
        let message = json!({ "type": "suite", "event": "started", "test_count": test_count });
        self.print_message(&message);
    }

    pub(crate) fn test_started(&mut self, test: &CollectedTest) {
        let name = test.desc.name.as_str();
        let message = json!({ "type": "test", "event": "started", "name": name });
        self.print_message(&message);
    }

    pub(crate) fn test_timed_out(&mut self, test: &CollectedTest) {
        let name = test.desc.name.as_str();
        let message = json!({ "type": "test", "event": "timeout", "name": name });
        self.print_message(&message);
    }

    pub(crate) fn test_finished(&mut self, test: &CollectedTest, completion: &TestCompletion) {
        let event;
        let name = test.desc.name.as_str();
        let mut maybe_message = None;
        let maybe_stdout = completion.stdout.as_deref().map(String::from_utf8_lossy);

        match completion.outcome {
            TestOutcome::Succeeded => {
                self.passed += 1;
                event = "ok";
            }
            TestOutcome::Failed { message } => {
                self.failed += 1;
                maybe_message = message;
                event = "failed";
            }
            TestOutcome::Ignored => {
                self.ignored += 1;
                maybe_message = test.desc.ignore_message.as_deref();
                event = "ignored";
            }
        };

        // This emits optional fields as `null`, instead of omitting them
        // completely as libtest does, but bootstrap can parse the result
        // either way.
        let json = json!({
            "type": "test",
            "event": event,
            "name": name,
            "message": maybe_message,
            "stdout": maybe_stdout,
        });

        self.print_message(&json);
    }

    pub(crate) fn suite_finished(&mut self) -> bool {
        let exec_time = self.suite_start.map(|start| (self.now() - start).as_secs_f64());
        let suite_passed = self.failed == 0;

        let event = if suite_passed { "ok" } else { "failed" };
        let message = json!({
            "type": "suite",
            "event": event,
            "passed": self.passed,
            "failed": self.failed,
            "ignored": self.ignored,
            // Compiletest doesn't run any benchmarks, but we still need to set this
            // field to 0 so that bootstrap's JSON parser can read our message.
            "measured": 0,
            "filtered_out": self.filtered_out,
            "exec_time": exec_time,
        });

        self.print_message(&message);
        suite_passed
    }
}
