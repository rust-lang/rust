//! This module encapsulates all of the code that interacts directly with
//! libtest, to execute the collected tests.
//!
//! This will hopefully make it easier to migrate away from libtest someday.

use std::borrow::Cow;
use std::sync::Arc;

use crate::common::{Config, TestPaths};

pub(crate) mod libtest;

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

/// Whether console output should be colored or not.
#[derive(Copy, Clone, Default, Debug)]
pub enum ColorConfig {
    #[default]
    AutoColor,
    AlwaysColor,
    NeverColor,
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

/// Whether test is expected to panic or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ShouldPanic {
    No,
    Yes,
}
