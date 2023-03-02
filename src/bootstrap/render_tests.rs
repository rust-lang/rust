//! This module renders the JSON output of libtest into a human-readable form, trying to be as
//! similar to libtest's native output as possible.
//!
//! This is needed because we need to use libtest in JSON mode to extract granluar information
//! about the executed tests. Doing so suppresses the human-readable output, and (compared to Cargo
//! and rustc) libtest doesn't include the rendered human-readable output as a JSON field. We had
//! to reimplement all the rendering logic in this module because of that.

use crate::builder::Builder;
use std::io::{BufRead, BufReader};
use std::process::{ChildStdout, Command, Stdio};
use std::time::Duration;

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
    let stdout = process.stdout.take().unwrap();
    let handle = std::thread::spawn(move || Renderer::new(stdout).render_all());

    let result = process.wait().unwrap();
    handle.join().expect("test formatter thread failed");

    if !result.success() && builder.is_verbose() {
        println!(
            "\n\ncommand did not execute successfully: {cmd:?}\n\
             expected success, got: {result}"
        );
    }

    result.success()
}

struct Renderer {
    stdout: BufReader<ChildStdout>,
    failures: Vec<TestOutcome>,
}

impl Renderer {
    fn new(stdout: ChildStdout) -> Self {
        Self { stdout: BufReader::new(stdout), failures: Vec::new() }
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

    fn render_test_outcome(&self, outcome: Outcome<'_>, test: &TestOutcome) {
        // TODO: add support for terse output
        self.render_test_outcome_verbose(outcome, test);
    }

    fn render_test_outcome_verbose(&self, outcome: Outcome<'_>, test: &TestOutcome) {
        if let Some(exec_time) = test.exec_time {
            println!(
                "test {} ... {outcome} (in {:.2?})",
                test.name,
                Duration::from_secs_f64(exec_time)
            );
        } else {
            println!("test {} ... {outcome}", test.name);
        }
    }

    fn render_suite_outcome(&self, outcome: Outcome<'_>, suite: &SuiteOutcome) {
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

        println!(
            "\ntest result: {outcome}. {} passed; {} failed; {} ignored; {} measured; \
             {} filtered out; finished in {:.2?}\n",
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
            }
            Message::Suite(SuiteMessage::Ok(outcome)) => {
                self.render_suite_outcome(Outcome::Ok, &outcome);
            }
            Message::Suite(SuiteMessage::Failed(outcome)) => {
                self.render_suite_outcome(Outcome::Failed, &outcome);
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
            Message::Test(TestMessage::Bench) => todo!("benchmarks are not supported yet"),
        }
    }
}

enum Outcome<'a> {
    Ok,
    Failed,
    Ignored { reason: Option<&'a str> },
}

impl std::fmt::Display for Outcome<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Ok => f.write_str("ok"),
            Outcome::Failed => f.write_str("FAILED"),
            Outcome::Ignored { reason: None } => f.write_str("ignored"),
            Outcome::Ignored { reason: Some(reason) } => write!(f, "ignored, {reason}"),
        }
    }
}

#[derive(serde_derive::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Message {
    Suite(SuiteMessage),
    Test(TestMessage),
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
    // Ignored messages:
    Bench,
    Started,
}

#[derive(serde_derive::Deserialize)]
struct TestOutcome {
    name: String,
    exec_time: Option<f64>,
    stdout: Option<String>,
    reason: Option<String>,
}
