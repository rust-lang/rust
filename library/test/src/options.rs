//! Enums denoting options for test execution.

/// Number of times to run a benchmarked function
#[derive(Clone, PartialEq, Eq)]
pub enum BenchMode {
    Auto,
    Single,
}

/// Whether test is expected to panic or not
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShouldPanic {
    No,
    Yes,
    YesWithMessage(&'static str),
}

/// Whether should console output be colored or not
#[derive(Copy, Clone, Default, Debug)]
pub enum ColorConfig {
    #[default]
    AutoColor,
    AlwaysColor,
    NeverColor,
}

/// Format of the test results output
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Verbose output
    Pretty,
    /// Quiet output
    #[default]
    Terse,
    /// JSON output
    Json,
    /// JUnit output
    Junit,
}

/// Whether ignored test should be run or not
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RunIgnored {
    Yes,
    No,
    /// Run only ignored tests
    Only,
}

#[derive(Clone, Copy)]
pub enum RunStrategy {
    /// Runs the test in the current process, and sends the result back over the
    /// supplied channel.
    InProcess,

    /// Spawns a subprocess to run the test, and sends the result back over the
    /// supplied channel. Requires `argv[0]` to exist and point to the binary
    /// that's currently running.
    SpawnPrimary,
}

/// Options for the test run defined by the caller (instead of CLI arguments).
/// In case we want to add other options as well, just add them in this struct.
#[derive(Copy, Clone, Debug)]
pub struct Options {
    pub display_output: bool,
    pub panic_abort: bool,
}

impl Options {
    pub fn new() -> Options {
        Options { display_output: false, panic_abort: false }
    }

    pub fn display_output(mut self, display_output: bool) -> Options {
        self.display_output = display_output;
        self
    }

    pub fn panic_abort(mut self, panic_abort: bool) -> Options {
        self.panic_abort = panic_abort;
        self
    }
}
