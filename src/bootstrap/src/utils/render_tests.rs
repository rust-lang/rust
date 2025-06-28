//! This module renders the JSON output of libtest into a human-readable form, trying to be as
//! similar to libtest's native output as possible.
//!
//! This is needed because we need to use libtest in JSON mode to extract granular information
//! about the executed tests. Doing so suppresses the human-readable output, and (compared to Cargo
//! and rustc) libtest doesn't include the rendered human-readable output as a JSON field. We had
//! to reimplement all the rendering logic in this module because of that.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{ChildStdout, Stdio};
use std::time::Duration;

use termcolor::{Color, ColorSpec, WriteColor};

use crate::core::builder::Builder;
use crate::utils::exec::BootstrapCommand;

const TERSE_TESTS_PER_LINE: usize = 88;

pub(crate) fn add_flags_and_try_run_tests(
    builder: &Builder<'_>,
    cmd: &mut BootstrapCommand,
) -> bool {
    if !cmd.get_args().any(|arg| arg == "--") {
        cmd.arg("--");
    }
    cmd.args(["-Z", "unstable-options", "--format", "json"]);

    try_run_tests(builder, cmd, false)
}

pub(crate) fn try_run_tests(
    builder: &Builder<'_>,
    cmd: &mut BootstrapCommand,
    stream: bool,
) -> bool {
    if builder.config.dry_run() {
        cmd.mark_as_executed();
        return true;
    }

    if !run_tests(builder, cmd, stream) {
        if builder.fail_fast {
            crate::exit!(1);
        } else {
            builder.config.exec_ctx().add_to_delay_failure(format!("{cmd:?}"));
            false
        }
    } else {
        true
    }
}

fn run_tests(builder: &Builder<'_>, cmd: &mut BootstrapCommand, stream: bool) -> bool {
    let cmd = cmd.as_command_mut();
    cmd.stdout(Stdio::piped());

    builder.verbose(|| println!("running: {cmd:?}"));

    let mut process = cmd.spawn().unwrap();

    // This runs until the stdout of the child is closed, which means the child exited. We don't
    // run this on another thread since the builder is not Sync.
    let renderer = Renderer::new(process.stdout.take().unwrap(), builder);
    if stream {
        renderer.stream_all();
    } else {
        renderer.render_all();
    }

    let result = process.wait_with_output().unwrap();
    if !result.status.success() && builder.is_verbose() {
        println!(
            "\n\ncommand did not execute successfully: {cmd:?}\n\
             expected success, got: {}",
            result.status
        );
    }

    result.status.success()
}

struct Renderer<'a> {
    stdout: BufReader<ChildStdout>,
    failures: Vec<TestOutcome>,
    benches: Vec<BenchOutcome>,
    builder: &'a Builder<'a>,
    tests_count: Option<usize>,
    executed_tests: usize,
    /// Number of tests that were skipped due to already being up-to-date
    /// (i.e. no relevant changes occurred since they last ran).
    up_to_date_tests: usize,
    ignored_tests: usize,
    terse_tests_in_line: usize,
    ci_latest_logged_percentage: f64,
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
            up_to_date_tests: 0,
            ignored_tests: 0,
            terse_tests_in_line: 0,
            ci_latest_logged_percentage: 0.0,
        }
    }

    fn render_all(mut self) {
        let mut line = Vec::new();
        loop {
            line.clear();
            match self.stdout.read_until(b'\n', &mut line) {
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(err) => panic!("failed to read output of test runner: {err}"),
            }
            if line.is_empty() {
                break;
            }

            match serde_json::from_slice(&line) {
                Ok(parsed) => self.render_message(parsed),
                Err(_err) => {
                    // Handle non-JSON output, for example when --nocapture is passed.
                    let mut stdout = std::io::stdout();
                    stdout.write_all(&line).unwrap();
                    let _ = stdout.flush();
                }
            }
        }

        if self.up_to_date_tests > 0 {
            let n = self.up_to_date_tests;
            let s = if n > 1 { "s" } else { "" };
            println!("help: ignored {n} up-to-date test{s}; use `--force-rerun` to prevent this\n");
        }
    }

    /// Renders the stdout characters one by one
    fn stream_all(mut self) {
        let mut buffer = [0; 1];
        loop {
            match self.stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(_) => {
                    let mut stdout = std::io::stdout();
                    stdout.write_all(&buffer).unwrap();
                    let _ = stdout.flush();
                }
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(err) => panic!("failed to read output of test runner: {err}"),
            }
        }
    }

    fn render_test_outcome(&mut self, outcome: Outcome<'_>, test: &TestOutcome) {
        self.executed_tests += 1;

        if let Outcome::Ignored { reason } = outcome {
            self.ignored_tests += 1;
            // Keep this in sync with the "up-to-date" ignore message inserted by compiletest.
            if reason == Some("up-to-date") {
                self.up_to_date_tests += 1;
            }
        }

        #[cfg(feature = "build-metrics")]
        self.builder.metrics.record_test(
            &test.name,
            match outcome {
                Outcome::Ok | Outcome::BenchOk => build_helper::metrics::TestOutcome::Passed,
                Outcome::Failed => build_helper::metrics::TestOutcome::Failed,
                Outcome::Ignored { reason } => build_helper::metrics::TestOutcome::Ignored {
                    ignore_reason: reason.map(|s| s.to_string()),
                },
            },
            self.builder,
        );

        if self.builder.config.verbose_tests {
            self.render_test_outcome_verbose(outcome, test);
        } else if self.builder.config.is_running_on_ci {
            self.render_test_outcome_ci(outcome, test);
        } else {
            self.render_test_outcome_terse(outcome, test);
        }
    }

    fn render_test_outcome_verbose(&self, outcome: Outcome<'_>, test: &TestOutcome) {
        print!("test {} ... ", test.name);
        self.builder.colored_stdout(|stdout| outcome.write_long(stdout)).unwrap();
        if let Some(exec_time) = test.exec_time {
            print!(" ({exec_time:.2?})");
        }
        println!();
    }

    fn render_test_outcome_terse(&mut self, outcome: Outcome<'_>, test: &TestOutcome) {
        if self.terse_tests_in_line != 0
            && self.terse_tests_in_line.is_multiple_of(TERSE_TESTS_PER_LINE)
        {
            if let Some(total) = self.tests_count {
                let total = total.to_string();
                let executed = format!("{:>width$}", self.executed_tests - 1, width = total.len());
                print!(" {executed}/{total}");
            }
            println!();
            self.terse_tests_in_line = 0;
        }

        self.terse_tests_in_line += 1;
        self.builder.colored_stdout(|stdout| outcome.write_short(stdout, &test.name)).unwrap();
        let _ = std::io::stdout().flush();
    }

    fn render_test_outcome_ci(&mut self, outcome: Outcome<'_>, test: &TestOutcome) {
        if let Some(total) = self.tests_count {
            let percent = self.executed_tests as f64 / total as f64;

            if self.ci_latest_logged_percentage + 0.10 < percent {
                let total = total.to_string();
                let executed = format!("{:>width$}", self.executed_tests, width = total.len());
                let pretty_percent = format!("{:.0}%", percent * 100.0);
                let passed_tests = self.executed_tests - (self.failures.len() + self.ignored_tests);
                println!(
                    "{:<4} -- {executed}/{total}, {:>total_indent$} passed, {} failed, {} ignored",
                    pretty_percent,
                    passed_tests,
                    self.failures.len(),
                    self.ignored_tests,
                    total_indent = total.len()
                );
                self.ci_latest_logged_percentage += 0.10;
            }
        }

        self.builder.colored_stdout(|stdout| outcome.write_ci(stdout, &test.name)).unwrap();
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
                if failure.stdout.is_some() || failure.message.is_some() {
                    println!("---- {} stdout ----", failure.name);
                    if let Some(stdout) = &failure.stdout {
                        println!("{stdout}");
                    }
                    if let Some(message) = &failure.message {
                        println!("NOTE: {message}");
                    }
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
                    format!("{:.2?}ns/iter", bench.median),
                    format!("+/- {:.2?}", bench.deviation),
                ));
            }

            let max_0 = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
            let max_1 = rows.iter().map(|r| r.1.len()).max().unwrap_or(0);
            let max_2 = rows.iter().map(|r| r.2.len()).max().unwrap_or(0);
            for row in &rows {
                println!("    {:<max_0$} {:>max_1$} {:>max_2$}", row.0, row.1, row.2);
            }
        }

        print!("\ntest result: ");
        self.builder.colored_stdout(|stdout| outcome.write_long(stdout)).unwrap();
        println!(
            ". {} passed; {} failed; {} ignored; {} measured; {} filtered out{time}\n",
            suite.passed,
            suite.failed,
            suite.ignored,
            suite.measured,
            suite.filtered_out,
            time = match suite.exec_time {
                Some(t) => format!("; finished in {:.2?}", Duration::from_secs_f64(t)),
                None => String::new(),
            }
        );
    }

    fn render_message(&mut self, message: Message) {
        match message {
            Message::Suite(SuiteMessage::Started { test_count }) => {
                println!("\nrunning {test_count} tests");
                self.benches = vec![];
                self.failures = vec![];
                self.ignored_tests = 0;
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
                    message: None,
                };
                self.render_test_outcome(Outcome::BenchOk, &fake_test_outcome);
                self.benches.push(outcome);
            }
            Message::Test(TestMessage::Ok(outcome)) => {
                self.render_test_outcome(Outcome::Ok, &outcome);
            }
            Message::Test(TestMessage::Ignored(outcome)) => {
                self.render_test_outcome(
                    Outcome::Ignored { reason: outcome.message.as_deref() },
                    &outcome,
                );
            }
            Message::Test(TestMessage::Failed(outcome)) => {
                self.render_test_outcome(Outcome::Failed, &outcome);
                self.failures.push(outcome);
            }
            Message::Test(TestMessage::Timeout { name }) => {
                println!("test {name} has been running for a long time");
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
    fn write_short(&self, writer: &mut dyn WriteColor, name: &str) -> Result<(), std::io::Error> {
        match self {
            Outcome::Ok => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Green)))?;
                write!(writer, ".")?;
            }
            Outcome::BenchOk => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Cyan)))?;
                write!(writer, "b")?;
            }
            Outcome::Failed => {
                // Put failed tests on their own line and include the test name, so that it's faster
                // to see which test failed without having to wait for them all to run.
                writeln!(writer)?;
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;
                writeln!(writer, "{name} ... F")?;
            }
            Outcome::Ignored { .. } => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Yellow)))?;
                write!(writer, "i")?;
            }
        }
        writer.reset()
    }

    fn write_long(&self, writer: &mut dyn WriteColor) -> Result<(), std::io::Error> {
        match self {
            Outcome::Ok => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Green)))?;
                write!(writer, "ok")?;
            }
            Outcome::BenchOk => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Cyan)))?;
                write!(writer, "benchmarked")?;
            }
            Outcome::Failed => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;
                write!(writer, "FAILED")?;
            }
            Outcome::Ignored { reason } => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Yellow)))?;
                write!(writer, "ignored")?;
                if let Some(reason) = reason {
                    write!(writer, ", {reason}")?;
                }
            }
        }
        writer.reset()
    }

    fn write_ci(&self, writer: &mut dyn WriteColor, name: &str) -> Result<(), std::io::Error> {
        match self {
            Outcome::Ok | Outcome::BenchOk | Outcome::Ignored { .. } => {}
            Outcome::Failed => {
                writer.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;
                writeln!(writer, "   {name} ... FAILED")?;
            }
        }
        writer.reset()
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
    /// The time it took to execute this test suite, or `None` if time measurement was not possible
    /// (e.g. due to running on wasm).
    exec_time: Option<f64>,
}

#[derive(serde_derive::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum TestMessage {
    Ok(TestOutcome),
    Failed(TestOutcome),
    Ignored(TestOutcome),
    Timeout { name: String },
    Started,
}

#[derive(serde_derive::Deserialize)]
struct BenchOutcome {
    name: String,
    median: f64,
    deviation: f64,
}

#[derive(serde_derive::Deserialize)]
struct TestOutcome {
    name: String,
    exec_time: Option<f64>,
    stdout: Option<String>,
    message: Option<String>,
}
