//! libtest formatting

use crate::{bench, test};
use std::{fs, io, path};

/// Test name padding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Padding {
    None,
    OnRight,
}

mod json;
pub use self::json::JsonFormatter;

mod pretty;
pub use self::pretty::PrettyFormatter;

mod terse;
pub use self::terse::TerseFormatter;

pub trait OutputFormatter {
    fn write_run_start(&mut self, test_count: usize) -> io::Result<()>;
    fn write_test_start(&mut self, desc: &test::Desc) -> io::Result<()>;
    fn write_timeout(&mut self, desc: &test::Desc) -> io::Result<()>;
    fn write_test_result(
        &mut self,
        desc: &test::Desc,
        result: &test::Result,
        stdout: &[u8],
    ) -> io::Result<()>;
    fn write_bench_result(
        &mut self,
        desc: &bench::Desc,
        result: &bench::Result,
        stdout: &[u8],
    ) -> io::Result<()>;
    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool>;
}

pub enum OutputLocation<T> {
    Pretty(Box<term::StdoutTerminal>),
    Raw(T),
}

impl<T: io::Write> io::Write for OutputLocation<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match *self {
            OutputLocation::Pretty(ref mut term) => term.write(buf),
            OutputLocation::Raw(ref mut stdout) => stdout.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match *self {
            OutputLocation::Pretty(ref mut term) => term.flush(),
            OutputLocation::Raw(ref mut stdout) => stdout.flush(),
        }
    }
}

#[derive(Debug)]
pub struct Options {
    pub logfile: Option<path::PathBuf>,
    pub display_output: bool,
}

impl Options {
    pub fn new() -> Self {
        Self {
            logfile: None,
            display_output: false,

        }

    }
}

pub struct ConsoleTestState {
    pub log_out: Option<fs::File>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub allowed_fail: usize,
    pub filtered_out: usize,
    pub measured: usize,
    pub metrics: bench::MetricMap,
    pub failures: Vec<(test::Desc, Vec<u8>)>,
    pub not_failures: Vec<(test::Desc, Vec<u8>)>,
    pub options: Options,
}

impl ConsoleTestState {
    pub fn new(options: Options) -> io::Result<ConsoleTestState> {
        let log_out = match options.logfile {
            Some(ref path) => Some(fs::File::create(path)?),
            None => None,
        };

        Ok(ConsoleTestState {
            log_out,
            total: 0,
            passed: 0,
            failed: 0,
            ignored: 0,
            allowed_fail: 0,
            filtered_out: 0,
            measured: 0,
            metrics: bench::MetricMap::new(),
            failures: Vec::new(),
            not_failures: Vec::new(),
            options,
        })
    }

    pub fn write_log<S: AsRef<str>>(&mut self, msg: S) -> io::Result<()> {
        use io::Write;
        let msg = msg.as_ref();
        match self.log_out {
            None => Ok(()),
            Some(ref mut o) => o.write_all(msg.as_bytes()),
        }
    }

    pub fn write_log_test_result(
        &mut self,
        test: &test::Desc,
        result: &test::Result,
    ) -> io::Result<()> {
        self.write_log(format!(
            "{} {}\n",
            match *result {
                test::Result::Ok => "ok".to_owned(),
                test::Result::Failed => "failed".to_owned(),
                test::Result::FailedMsg(ref msg) => format!("failed: {}", msg),
                test::Result::Ignored => "ignored".to_owned(),
                test::Result::AllowedFail => "failed (allowed)".to_owned(),
            },
            test.name
        ))
    }

    pub fn write_log_bench_result(
        &mut self,
        test: &bench::Desc,
        result: &bench::Result,
    ) -> io::Result<()> {
        self.write_log(format!(
            "{} {}\n",
            match *result {
                bench::Result::Ok(ref bs) => format!("{}", bs),
                bench::Result::Failed => "failed".to_owned(),
            },
            test.name
        ))
    }

    pub fn current_test_count(&self) -> usize {
        self.passed + self.failed + self.ignored + self.measured + self.allowed_fail
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ColorConfig {
    AutoColor,
    AlwaysColor,
    NeverColor,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Pretty,
    Terse,
    Json,
}
