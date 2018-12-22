//! libtest utilities

#![feature(uniform_paths)]

pub mod bench;
pub mod format;
pub mod test;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShouldPanic {
    No,
    Yes,
    YesWithMessage(&'static str),
}

/// Seconds after which a test-timed-out warning is emitted
pub const TEST_WARN_TIMEOUT_S: u64 = 60;

/// Insert a '\n' after 100 tests in quiet mode
pub const QUIET_MODE_MAX_COLUMN: usize = 100;

mod isatty;
pub use isatty::stdout_isatty;

mod get_concurrency;
pub use get_concurrency::get_concurrency;
