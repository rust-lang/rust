//! This module renders the JSON output of libtest into a human-readable form, trying to be as
//! similar to libtest's native output as possible.
//!
//! This is needed because we need to use libtest in JSON mode to extract granluar information
//! about the executed tests. Doing so suppresses the human-readable output, and (compared to Cargo
//! and rustc) libtest doesn't include the rendered human-readable output as a JSON field. We had
//! to reimplement all the rendering logic in this module because of that.

use crate::builder::Builder;
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdout, Command, Stdio};
use std::time::Duration;
use yansi_term::Color;

const TERSE_TESTS_PER_LINE: usize = 88;

pub(crate) fn try_run_tests(builder: &Builder<'_>, cmd: &mut Command) -> bool {
    if builder.config.dry_run() {
        return true;
    }

    if !run_tests(builder, cmd) {
        if builder.fail_fast {
            crate::detail_exit(1);
        } else {
            let mut failures = builder.delayed_failures.borrow_mut();
            failures.push(format!("{cmd:?}"));
            false
        }
    } else {
        true
    }
}

fn run_tests(builder: &Builder<'_>, cmd: &mut Command) -> bool {
    cmd.stdout(Stdio::piped());

    builder.verbose(&format!("running: {cmd:?}"));

    let mut process = cmd.spawn().unwrap();

    // This runs until the stdout of the child is closed, which means the child exited. We don't
    // run this on another thread since the builder is not Sync.
    Renderer::new(process.stdout.take().unwrap(), builder).render_all();

    let result = process.wait().unwrap();
    if !result.success() && builder.is_verbose() {
        println!(
            "\n\ncommand did not execute successfully: {cmd:?}\n\
             expected success, got: {result}"
        );
    }

    result.success()
}

struct Renderer<'a> {
    stdout: BufReader<ChildStdout>,
    failures: Vec<TestOutcome>,
    benches: Vec<BenchOutcome>,
    builder: &'a Builder<'a>,
    tests_count: Option<usize>,
    executed_tests: usize,
    terse_tests_in_line: usize,
}

impl<'a> Renderer<'a> {
    fn new(stdout: ChildStdout, builder: &'a Builder<'a>) -> Self {
        Self {
            stdout: BufReader::new(stdout),
            benches: Vec::new(),
            failures: Vec::new(),
            builder,
            tests_count: None,
            executed_tests: 0,
            terse_tests_in_line: 0,
        }
    }

    fn render_all(mut self) {
        let mut line = String::new();
        loop {
            line.clear();
            match self.stdout.read_line(&mut line) {
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(err) => panic!("failed to read output of test runner: {err}"),
            }
            if line.is_empty() {
                break;
            }

            self.render_message(match serde_json::from_str(&line) {
                Ok(parsed) => parsed,
                Err(err) => {
                    panic!("failed to parse libtest json output; error: {err}, line: {line:?}");
                }
            });
        }
    }

    fn render_test_outcome(&mut self, outcome: Outcome<'_>, test: &TestOutcome) {
        self.executed_tests += 1;

        #[cfg(feature = "build-metrics")]
        self.builder.metrics.record_test(
            &test.name,
            match outcome {
                Outcome::Ok | Outcome::BenchOk => crate::metrics::TestOutcome::Passed,
                Outcome::Failed => crate::metrics::TestOutcome::Failed,
                Outcome::Ignored { reason } => crate::metrics::TestOutcome::Ignored {
                    ignore_reason: reason.map(|s| s.to_string()),
                },
            },
        );

        if self.builder.config.verbose_tests {
            self.render_test_outcome_verbose(outcome, test);
        } else {
            self.render_test_outcome_terse(outcome, test);
        }
    }

    fn render_test_outcome_verbose(&self, outcome: Outcome<'_>, test: &TestOutcome) {
        if let Some(exec_time) = test.exec_time {
            println!(
                "test {} ... {} (in {:.2?})",
                test.name,
                outcome.long(self.builder),
                Duration::from_secs_f64(exec_time)
            );
        } else {
            println!("test {} ... {}", test.name, outcome.long(self.builder));
        }
    }

    fn render_test_outcome_terse(&mut self, outcome: Outcome<'_>, _: &TestOutcome) {
        if self.terse_tests_in_line != 0 && self.terse_tests_in_line % TERSE_TESTS_PER_LINE == 0 {
            if let Some(total) = self.tests_count {
                let total = total.to_string();
                let executed = format!("{:>width$}", self.executed_tests - 1, width = total.len());
                print!(" {executed}/{total}");
            }
            println!();
            self.terse_tests_in_line = 0;
        }

        self.terse_tests_in_line += 1;
        print!("{}", outcome.short(self.builder));
        let _ = std::io::stdout().flush();
    }

    fn render_suite_outcome(&self, outcome: Outcome<'_>, suite: &SuiteOutcome) {
        // The terse output doesn't end with a newline, so we need to add it ourselves.
        if !self.builder.config.verbose_tests {
            println!();
        }

        if !self.failures.is_empty() {
            println!("\nfailures:\n");
            for failure in &self.failures {
                if let Some(stdout) = &failure.stdout {
                    println!("---- {} stdout ----", failure.name);
                    println!("{stdout}");
                }
            }

            println!("\nfailures:");
            for failure in &self.failures {
                println!("    {}", failure.name);
            }
        }

        if !self.benches.is_empty() {
            println!("\nbenchmarks:");

            let mut rows = Vec::new();
            for bench in &self.benches {
                rows.push((
                    &bench.name,
                    format!("{:.2?}/iter", Duration::from_nanos(bench.median)),
                    format!("+/- {:.2?}", Duration::from_nanos(bench.deviation)),
                ));
            }

            let max_0 = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
            let max_1 = rows.iter().map(|r| r.1.len()).max().unwrap_or(0);
            let max_2 = rows.iter().map(|r| r.2.len()).max().unwrap_or(0);
            for row in &rows {
                println!("    {:<max_0$} {:>max_1$} {:>max_2$}", row.0, row.1, row.2);
            }
        }

        println!(
            "\ntest result: {}. {} passed; {} failed; {} ignored; {} measured; \
             {} filtered out; finished in {:.2?}\n",
            outcome.long(self.builder),
            suite.passed,
            suite.failed,
            suite.ignored,
            suite.measured,
            suite.filtered_out,
            Duration::from_secs_f64(suite.exec_time)
        );
    }

    fn render_message(&mut self, message: Message) {
        match message {
            Message::Suite(SuiteMessage::Started { test_count }) => {
                println!("\nrunning {test_count} tests");
                self.executed_tests = 0;
                self.terse_tests_in_line = 0;
                self.tests_count = Some(test_count);
            }
            Message::Suite(SuiteMessage::Ok(outcome)) => {
                self.render_suite_outcome(Outcome::Ok, &outcome);
            }
            Message::Suite(SuiteMessage::Failed(outcome)) => {
                self.render_suite_outcome(Outcome::Failed, &outcome);
            }
            Message::Bench(outcome) => {
                // The formatting for benchmarks doesn't replicate 1:1 the formatting libtest
                // outputs, mostly because libtest's formatting is broken in terse mode, which is
                // the default used by our monorepo. We use a different formatting instead:
                // successful benchmarks are just showed as "benchmarked"/"b", and the details are
                // outputted at the bottom like failures.
                let fake_test_outcome = TestOutcome {
                    name: outcome.name.clone(),
                    exec_time: None,
                    stdout: None,
                    reason: None,
                };
                self.render_test_outcome(Outcome::BenchOk, &fake_test_outcome);
                self.benches.push(outcome);
            }
            Message::Test(TestMessage::Ok(outcome)) => {
                self.render_test_outcome(Outcome::Ok, &outcome);
            }
            Message::Test(TestMessage::Ignored(outcome)) => {
                self.render_test_outcome(
                    Outcome::Ignored { reason: outcome.reason.as_deref() },
                    &outcome,
                );
            }
            Message::Test(TestMessage::Failed(outcome)) => {
                self.render_test_outcome(Outcome::Failed, &outcome);
                self.failures.push(outcome);
            }
            Message::Test(TestMessage::Started) => {} // Not useful
        }
    }
}

enum Outcome<'a> {
    Ok,
    BenchOk,
    Failed,
    Ignored { reason: Option<&'a str> },
}

impl Outcome<'_> {
    fn short(&self, builder: &Builder<'_>) -> String {
        match self {
            Outcome::Ok => builder.color_for_stdout(Color::Green, "."),
            Outcome::BenchOk => builder.color_for_stdout(Color::Cyan, "b"),
            Outcome::Failed => builder.color_for_stdout(Color::Red, "F"),
            Outcome::Ignored { .. } => builder.color_for_stdout(Color::Yellow, "i"),
        }
    }

    fn long(&self, builder: &Builder<'_>) -> String {
        match self {
            Outcome::Ok => builder.color_for_stdout(Color::Green, "ok"),
            Outcome::BenchOk => builder.color_for_stdout(Color::Cyan, "benchmarked"),
            Outcome::Failed => builder.color_for_stdout(Color::Red, "FAILED"),
            Outcome::Ignored { reason: None } => builder.color_for_stdout(Color::Yellow, "ignored"),
            Outcome::Ignored { reason: Some(reason) } => {
                builder.color_for_stdout(Color::Yellow, &format!("ignored, {reason}"))
            }
        }
    }
}

#[derive(serde_derive::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Message {
    Suite(SuiteMessage),
    Test(TestMessage),
    Bench(BenchOutcome),
}

#[derive(serde_derive::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum SuiteMessage {
    Ok(SuiteOutcome),
    Failed(SuiteOutcome),
    Started { test_count: usize },
}

#[derive(serde_derive::Deserialize)]
struct SuiteOutcome {
    passed: usize,
    failed: usize,
    ignored: usize,
    measured: usize,
    filtered_out: usize,
    exec_time: f64,
}

#[derive(serde_derive::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum TestMessage {
    Ok(TestOutcome),
    Failed(TestOutcome),
    Ignored(TestOutcome),
    Started,
}

#[derive(serde_derive::Deserialize)]
struct BenchOutcome {
    name: String,
    median: u64,
    deviation: u64,
}

#[derive(serde_derive::Deserialize)]
struct TestOutcome {
    name: String,
    exec_time: Option<f64>,
    stdout: Option<String>,
    reason: Option<String>,
}
