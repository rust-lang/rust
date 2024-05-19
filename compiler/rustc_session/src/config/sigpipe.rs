//! NOTE: Keep these constants in sync with `library/std/src/sys/pal/unix/mod.rs`!

/// The default value if `-Zon-broken-pipe=...` is not specified. This resolves
/// to `SIG_IGN` in `library/std/src/sys/pal/unix/mod.rs`.
///
/// Note that `SIG_IGN` has been the Rust default since 2014. See
/// <https://github.com/rust-lang/rust/issues/62569>.
#[allow(dead_code)]
pub const DEFAULT: u8 = 0;

/// Do not touch `SIGPIPE`. Use whatever the parent process uses.
#[allow(dead_code)]
pub const INHERIT: u8 = 1;

/// Change `SIGPIPE` to `SIG_IGN` so that failed writes results in `EPIPE`
/// that are eventually converted to `ErrorKind::BrokenPipe`.
#[allow(dead_code)]
pub const SIG_IGN: u8 = 2;

/// Change `SIGPIPE` to `SIG_DFL` so that the process is killed when trying
/// to write to a closed pipe. This is usually the desired behavior for CLI
/// apps that produce textual output that you want to pipe to other programs
/// such as `head -n 1`.
#[allow(dead_code)]
pub const SIG_DFL: u8 = 3;
