//! Common types used by `libtest`.

use std::borrow::Cow;
use std::fmt;
use std::sync::mpsc::Sender;

pub use NamePadding::*;
pub use TestFn::*;
pub use TestName::*;

use super::bench::Bencher;
use super::event::CompletedTest;
use super::{__rust_begin_short_backtrace, options};

/// Type of the test according to the [Rust book](https://doc.rust-lang.org/cargo/guide/tests.html)
/// conventions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TestType {
    /// Unit-tests are expected to be in the `src` folder of the crate.
    UnitTest,
    /// Integration-style tests are expected to be in the `tests` folder of the crate.
    IntegrationTest,
    /// Doctests are created by the `librustdoc` manually, so it's a different type of test.
    DocTest,
    /// Tests for the sources that don't follow the project layout convention
    /// (e.g. tests in raw `main.rs` compiled by calling `rustc --test` directly).
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NamePadding {
    PadNone,
    PadOnRight,
}

// The name of a test. By convention this follows the rules for rust
// paths; i.e., it should be a series of identifiers separated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TestName {
    StaticTestName(&'static str),
    DynTestName(String),
    AlignedTestName(Cow<'static, str>, NamePadding),
}

impl TestName {
    pub fn as_slice(&self) -> &str {
        match *self {
            StaticTestName(s) => s,
            DynTestName(ref s) => s,
            AlignedTestName(ref s, _) => s,
        }
    }

    pub fn padding(&self) -> NamePadding {
        match self {
            &AlignedTestName(_, p) => p,
            _ => PadNone,
        }
    }

    pub fn with_padding(&self, padding: NamePadding) -> TestName {
        let name = match *self {
            TestName::StaticTestName(name) => Cow::Borrowed(name),
            TestName::DynTestName(ref name) => Cow::Owned(name.clone()),
            TestName::AlignedTestName(ref name, _) => name.clone(),
        };

        TestName::AlignedTestName(name, padding)
    }
}
impl fmt::Display for TestName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), f)
    }
}

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function panics or returns Result::Err
// then the test fails. We may need to come up with a more clever
// definition of test in order to support isolation of tests into
// threads.
pub enum TestFn {
    StaticTestFn(fn() -> Result<(), String>),
    StaticBenchFn(fn(&mut Bencher) -> Result<(), String>),
    StaticBenchAsTestFn(fn(&mut Bencher) -> Result<(), String>),
    DynTestFn(Box<dyn FnOnce() -> Result<(), String> + Send>),
    DynBenchFn(Box<dyn Fn(&mut Bencher) -> Result<(), String> + Send>),
    DynBenchAsTestFn(Box<dyn Fn(&mut Bencher) -> Result<(), String> + Send>),
}

impl TestFn {
    pub fn padding(&self) -> NamePadding {
        match *self {
            StaticTestFn(..) => PadNone,
            StaticBenchFn(..) => PadOnRight,
            StaticBenchAsTestFn(..) => PadNone,
            DynTestFn(..) => PadNone,
            DynBenchFn(..) => PadOnRight,
            DynBenchAsTestFn(..) => PadNone,
        }
    }

    pub(crate) fn into_runnable(self) -> Runnable {
        match self {
            StaticTestFn(f) => Runnable::Test(RunnableTest::Static(f)),
            StaticBenchFn(f) => Runnable::Bench(RunnableBench::Static(f)),
            StaticBenchAsTestFn(f) => Runnable::Test(RunnableTest::StaticBenchAsTest(f)),
            DynTestFn(f) => Runnable::Test(RunnableTest::Dynamic(f)),
            DynBenchFn(f) => Runnable::Bench(RunnableBench::Dynamic(f)),
            DynBenchAsTestFn(f) => Runnable::Test(RunnableTest::DynamicBenchAsTest(f)),
        }
    }
}

impl fmt::Debug for TestFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            StaticTestFn(..) => "StaticTestFn(..)",
            StaticBenchFn(..) => "StaticBenchFn(..)",
            StaticBenchAsTestFn(..) => "StaticBenchAsTestFn(..)",
            DynTestFn(..) => "DynTestFn(..)",
            DynBenchFn(..) => "DynBenchFn(..)",
            DynBenchAsTestFn(..) => "DynBenchAsTestFn(..)",
        })
    }
}

pub(crate) enum Runnable {
    Test(RunnableTest),
    Bench(RunnableBench),
}

pub(crate) enum RunnableTest {
    Static(fn() -> Result<(), String>),
    Dynamic(Box<dyn FnOnce() -> Result<(), String> + Send>),
    StaticBenchAsTest(fn(&mut Bencher) -> Result<(), String>),
    DynamicBenchAsTest(Box<dyn Fn(&mut Bencher) -> Result<(), String> + Send>),
}

impl RunnableTest {
    pub(crate) fn run(self) -> Result<(), String> {
        match self {
            RunnableTest::Static(f) => __rust_begin_short_backtrace(f),
            RunnableTest::Dynamic(f) => __rust_begin_short_backtrace(f),
            RunnableTest::StaticBenchAsTest(f) => {
                crate::bench::run_once(|b| __rust_begin_short_backtrace(|| f(b)))
            }
            RunnableTest::DynamicBenchAsTest(f) => {
                crate::bench::run_once(|b| __rust_begin_short_backtrace(|| f(b)))
            }
        }
    }

    pub(crate) fn is_dynamic(&self) -> bool {
        match self {
            RunnableTest::Static(_) => false,
            RunnableTest::StaticBenchAsTest(_) => false,
            RunnableTest::Dynamic(_) => true,
            RunnableTest::DynamicBenchAsTest(_) => true,
        }
    }
}

pub(crate) enum RunnableBench {
    Static(fn(&mut Bencher) -> Result<(), String>),
    Dynamic(Box<dyn Fn(&mut Bencher) -> Result<(), String> + Send>),
}

impl RunnableBench {
    pub(crate) fn run(
        self,
        id: TestId,
        desc: &TestDesc,
        monitor_ch: &Sender<CompletedTest>,
        nocapture: bool,
    ) {
        match self {
            RunnableBench::Static(f) => {
                crate::bench::benchmark(id, desc.clone(), monitor_ch.clone(), nocapture, f)
            }
            RunnableBench::Dynamic(f) => {
                crate::bench::benchmark(id, desc.clone(), monitor_ch.clone(), nocapture, f)
            }
        }
    }
}

// A unique integer associated with each test.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TestId(pub usize);

// The definition of a single test. A test runner will run a list of
// these.
#[derive(Clone, Debug)]
pub struct TestDesc {
    pub name: TestName,
    pub ignore: bool,
    pub ignore_message: Option<&'static str>,
    pub source_file: &'static str,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub should_panic: options::ShouldPanic,
    pub compile_fail: bool,
    pub no_run: bool,
    pub test_type: TestType,
}

impl TestDesc {
    pub fn padded_name(&self, column_count: usize, align: NamePadding) -> String {
        let mut name = String::from(self.name.as_slice());
        let fill = column_count.saturating_sub(name.len());
        let pad = " ".repeat(fill);
        match align {
            PadNone => name,
            PadOnRight => {
                name.push_str(&pad);
                name
            }
        }
    }

    /// Returns None for ignored test or tests that are just run, otherwise returns a description of the type of test.
    /// Descriptions include "should panic", "compile fail" and "compile".
    pub fn test_mode(&self) -> Option<&'static str> {
        if self.ignore {
            return None;
        }
        match self.should_panic {
            options::ShouldPanic::Yes | options::ShouldPanic::YesWithMessage(_) => {
                return Some("should panic");
            }
            options::ShouldPanic::No => {}
        }
        if self.compile_fail {
            return Some("compile fail");
        }
        if self.no_run {
            return Some("compile");
        }
        None
    }
}

#[derive(Debug)]
pub struct TestDescAndFn {
    pub desc: TestDesc,
    pub testfn: TestFn,
}

impl TestDescAndFn {
    pub const fn new_doctest(
        test_name: &'static str,
        ignore: bool,
        source_file: &'static str,
        start_line: usize,
        no_run: bool,
        should_panic: bool,
        testfn: TestFn,
    ) -> Self {
        Self {
            desc: TestDesc {
                name: StaticTestName(test_name),
                ignore,
                ignore_message: None,
                source_file,
                start_line,
                start_col: 0,
                end_line: 0,
                end_col: 0,
                compile_fail: false,
                no_run,
                should_panic: if should_panic {
                    options::ShouldPanic::Yes
                } else {
                    options::ShouldPanic::No
                },
                test_type: TestType::DocTest,
            },
            testfn,
        }
    }
}
