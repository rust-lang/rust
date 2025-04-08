//! This module encapsulates all of the code that interacts directly with
//! libtest, to execute the collected tests.
//!
//! This will hopefully make it easier to migrate away from libtest someday.

use std::borrow::Cow;
use std::io;
use std::sync::Arc;

use crate::common::{Config, TestPaths};

/// Delegates to libtest to run the list of collected tests.
///
/// Returns `Ok(true)` if all tests passed, or `Ok(false)` if one or more tests failed.
pub(crate) fn execute_tests(config: &Config, tests: Vec<CollectedTest>) -> io::Result<bool> {
    let opts = test_opts(config);
    let tests = tests.into_iter().map(|t| t.into_libtest()).collect::<Vec<_>>();

    test::run_tests_console(&opts, tests)
}

/// Information needed to create a `test::TestDescAndFn`.
pub(crate) struct CollectedTest {
    pub(crate) desc: CollectedTestDesc,
    pub(crate) config: Arc<Config>,
    pub(crate) testpaths: TestPaths,
    pub(crate) revision: Option<String>,
}

/// Information needed to create a `test::TestDesc`.
pub(crate) struct CollectedTestDesc {
    pub(crate) name: String,
    pub(crate) ignore: bool,
    pub(crate) ignore_message: Option<Cow<'static, str>>,
    pub(crate) should_panic: ShouldPanic,
}

impl CollectedTest {
    fn into_libtest(self) -> test::TestDescAndFn {
        let Self { desc, config, testpaths, revision } = self;
        let CollectedTestDesc { name, ignore, ignore_message, should_panic } = desc;

        // Libtest requires the ignore message to be a &'static str, so we might
        // have to leak memory to create it. This is fine, as we only do so once
        // per test, so the leak won't grow indefinitely.
        let ignore_message = ignore_message.map(|msg| match msg {
            Cow::Borrowed(s) => s,
            Cow::Owned(s) => &*String::leak(s),
        });

        let desc = test::TestDesc {
            name: test::DynTestName(name),
            ignore,
            ignore_message,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: should_panic.to_libtest(),
            compile_fail: false,
            no_run: false,
            test_type: test::TestType::Unknown,
        };

        // This closure is invoked when libtest returns control to compiletest
        // to execute the test.
        let testfn = test::DynTestFn(Box::new(move || {
            crate::runtest::run(config, &testpaths, revision.as_deref());
            Ok(())
        }));

        test::TestDescAndFn { desc, testfn }
    }
}

/// Whether console output should be colored or not.
#[derive(Copy, Clone, Default, Debug)]
pub enum ColorConfig {
    #[default]
    AutoColor,
    AlwaysColor,
    NeverColor,
}

impl ColorConfig {
    fn to_libtest(self) -> test::ColorConfig {
        match self {
            Self::AutoColor => test::ColorConfig::AutoColor,
            Self::AlwaysColor => test::ColorConfig::AlwaysColor,
            Self::NeverColor => test::ColorConfig::NeverColor,
        }
    }
}

/// Format of the test results output.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Verbose output
    Pretty,
    /// Quiet output
    #[default]
    Terse,
    /// JSON output
    Json,
}

impl OutputFormat {
    fn to_libtest(self) -> test::OutputFormat {
        match self {
            Self::Pretty => test::OutputFormat::Pretty,
            Self::Terse => test::OutputFormat::Terse,
            Self::Json => test::OutputFormat::Json,
        }
    }
}

/// Whether test is expected to panic or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ShouldPanic {
    No,
    Yes,
}

impl ShouldPanic {
    fn to_libtest(self) -> test::ShouldPanic {
        match self {
            Self::No => test::ShouldPanic::No,
            Self::Yes => test::ShouldPanic::Yes,
        }
    }
}

fn test_opts(config: &Config) -> test::TestOpts {
    test::TestOpts {
        exclude_should_panic: false,
        filters: config.filters.clone(),
        filter_exact: config.filter_exact,
        run_ignored: if config.run_ignored { test::RunIgnored::Yes } else { test::RunIgnored::No },
        format: config.format.to_libtest(),
        logfile: None,
        run_tests: true,
        bench_benchmarks: true,
        nocapture: config.nocapture,
        color: config.color.to_libtest(),
        shuffle: false,
        shuffle_seed: None,
        test_threads: None,
        skip: config.skip.clone(),
        list: false,
        options: test::Options::new(),
        time_options: None,
        force_run_in_process: false,
        fail_fast: config.fail_fast,
    }
}
