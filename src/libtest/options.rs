//! Enums denoting options for test execution.

/// Whether to execute tests concurrently or not
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Concurrent {
    Yes,
    No,
}

#[derive(Clone, PartialEq, Eq)]
pub enum BenchMode {
    Auto,
    Single,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShouldPanic {
    No,
    Yes,
    YesWithMessage(&'static str),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RunIgnored {
    Yes,
    No,
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

/// In case we want to add other options as well, just add them in this struct.
#[derive(Copy, Clone, Debug)]
pub struct Options {
    pub display_output: bool,
    pub panic_abort: bool,
}

impl Options {
    pub fn new() -> Options {
        Options {
            display_output: false,
            panic_abort: false,
        }
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
