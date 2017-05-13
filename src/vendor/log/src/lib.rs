// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A lightweight logging facade.
//!
//! A logging facade provides a single logging API that abstracts over the
//! actual logging implementation. Libraries can use the logging API provided
//! by this crate, and the consumer of those libraries can choose the logging
//! framework that is most suitable for its use case.
//!
//! If no logging implementation is selected, the facade falls back to a "noop"
//! implementation that ignores all log messages. The overhead in this case
//! is very small - just an integer load, comparison and jump.
//!
//! A log request consists of a target, a level, and a body. A target is a
//! string which defaults to the module path of the location of the log
//! request, though that default may be overridden. Logger implementations
//! typically use the target to filter requests based on some user
//! configuration.
//!
//! # Use
//!
//! ## In libraries
//!
//! Libraries should link only to the `log` crate, and use the provided
//! macros to log whatever information will be useful to downstream consumers.
//!
//! ### Examples
//!
//! ```rust
//! # #![allow(unstable)]
//! #[macro_use]
//! extern crate log;
//!
//! # #[derive(Debug)] pub struct Yak(String);
//! # impl Yak { fn shave(&self, _: u32) {} }
//! # fn find_a_razor() -> Result<u32, u32> { Ok(1) }
//! pub fn shave_the_yak(yak: &Yak) {
//!     info!(target: "yak_events", "Commencing yak shaving for {:?}", yak);
//!
//!     loop {
//!         match find_a_razor() {
//!             Ok(razor) => {
//!                 info!("Razor located: {}", razor);
//!                 yak.shave(razor);
//!                 break;
//!             }
//!             Err(err) => {
//!                 warn!("Unable to locate a razor: {}, retrying", err);
//!             }
//!         }
//!     }
//! }
//! # fn main() {}
//! ```
//!
//! ## In executables
//!
//! Executables should choose a logging framework and initialize it early in the
//! runtime of the program. Logging frameworks will typically include a
//! function to do this. Any log messages generated before the framework is
//! initialized will be ignored.
//!
//! The executable itself may use the `log` crate to log as well.
//!
//! ### Warning
//!
//! The logging system may only be initialized once.
//!
//! ### Examples
//!
//! ```rust,ignore
//! #[macro_use]
//! extern crate log;
//! extern crate my_logger;
//!
//! fn main() {
//!     my_logger::init();
//!
//!     info!("starting up");
//!
//!     // ...
//! }
//! ```
//!
//! # Logger implementations
//!
//! Loggers implement the `Log` trait. Here's a very basic example that simply
//! logs all messages at the `Error`, `Warn` or `Info` levels to stdout:
//!
//! ```rust
//! extern crate log;
//!
//! use log::{LogRecord, LogLevel, LogMetadata};
//!
//! struct SimpleLogger;
//!
//! impl log::Log for SimpleLogger {
//!     fn enabled(&self, metadata: &LogMetadata) -> bool {
//!         metadata.level() <= LogLevel::Info
//!     }
//!
//!     fn log(&self, record: &LogRecord) {
//!         if self.enabled(record.metadata()) {
//!             println!("{} - {}", record.level(), record.args());
//!         }
//!     }
//! }
//!
//! # fn main() {}
//! ```
//!
//! Loggers are installed by calling the `set_logger` function. It takes a
//! closure which is provided a `MaxLogLevel` token and returns a `Log` trait
//! object. The `MaxLogLevel` token controls the global maximum log level. The
//! logging facade uses this as an optimization to improve performance of log
//! messages at levels that are disabled. In the case of our example logger,
//! we'll want to set the maximum log level to `Info`, since we ignore any
//! `Debug` or `Trace` level log messages. A logging framework should provide a
//! function that wraps a call to `set_logger`, handling initialization of the
//! logger:
//!
//! ```rust
//! # extern crate log;
//! # use log::{LogLevel, LogLevelFilter, SetLoggerError, LogMetadata};
//! # struct SimpleLogger;
//! # impl log::Log for SimpleLogger {
//! #   fn enabled(&self, _: &LogMetadata) -> bool { false }
//! #   fn log(&self, _: &log::LogRecord) {}
//! # }
//! # fn main() {}
//! # #[cfg(feature = "use_std")]
//! pub fn init() -> Result<(), SetLoggerError> {
//!     log::set_logger(|max_log_level| {
//!         max_log_level.set(LogLevelFilter::Info);
//!         Box::new(SimpleLogger)
//!     })
//! }
//! ```
//!
//! # Use with `no_std`
//!
//! To use the `log` crate without depending on `libstd`, you need to specify
//! `default-features = false` when specifying the dependency in `Cargo.toml`.
//! This makes no difference to libraries using `log` since the logging API
//! remains the same. However executables will need to use the `set_logger_raw`
//! function to initialize a logger and the `shutdown_logger_raw` function to
//! shut down the global logger before exiting:
//!
//! ```rust
//! # extern crate log;
//! # use log::{LogLevel, LogLevelFilter, SetLoggerError, ShutdownLoggerError,
//! #           LogMetadata};
//! # struct SimpleLogger;
//! # impl log::Log for SimpleLogger {
//! #   fn enabled(&self, _: &LogMetadata) -> bool { false }
//! #   fn log(&self, _: &log::LogRecord) {}
//! # }
//! # impl SimpleLogger {
//! #   fn flush(&self) {}
//! # }
//! # fn main() {}
//! pub fn init() -> Result<(), SetLoggerError> {
//!     unsafe {
//!         log::set_logger_raw(|max_log_level| {
//!             static LOGGER: SimpleLogger = SimpleLogger;
//!             max_log_level.set(LogLevelFilter::Info);
//!             &SimpleLogger
//!         })
//!     }
//! }
//! pub fn shutdown() -> Result<(), ShutdownLoggerError> {
//!     log::shutdown_logger_raw().map(|logger| {
//!         let logger = unsafe { &*(logger as *const SimpleLogger) };
//!         logger.flush();
//!     })
//! }
//! ```

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://www.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/log/")]
#![warn(missing_docs)]
#![cfg_attr(feature = "nightly", feature(panic_handler))]

#![cfg_attr(not(feature = "use_std"), no_std)]

#[cfg(not(feature = "use_std"))]
extern crate core as std;

use std::cmp;
#[cfg(feature = "use_std")]
use std::error;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
#[macro_use]
mod macros;

// The setup here is a bit weird to make shutdown_logger_raw work.
//
// There are four different states that we care about: the logger's
// uninitialized, the logger's initializing (set_logger's been called but
// LOGGER hasn't actually been set yet), the logger's active, or the logger is
// shut down after calling shutdown_logger_raw.
//
// The LOGGER static holds a pointer to the global logger. It is protected by
// the STATE static which determines whether LOGGER has been initialized yet.
//
// The shutdown_logger_raw routine needs to make sure that no threads are
// actively logging before it returns. The number of actively logging threads is
// tracked in the REFCOUNT static. The routine first sets STATE back to
// INITIALIZING. All logging calls past that point will immediately return
// without accessing the logger. At that point, the at_exit routine just waits
// for the refcount to reach 0 before deallocating the logger. Note that the
// refcount does not necessarily monotonically decrease at this point, as new
// log calls still increment and decrement it, but the interval in between is
// small enough that the wait is really just for the active log calls to finish.

static mut LOGGER: *const Log = &NopLogger;
static STATE: AtomicUsize = ATOMIC_USIZE_INIT;
static REFCOUNT: AtomicUsize = ATOMIC_USIZE_INIT;

const UNINITIALIZED: usize = 0;
const INITIALIZING: usize = 1;
const INITIALIZED: usize = 2;

static MAX_LOG_LEVEL_FILTER: AtomicUsize = ATOMIC_USIZE_INIT;

static LOG_LEVEL_NAMES: [&'static str; 6] = ["OFF", "ERROR", "WARN", "INFO",
                                             "DEBUG", "TRACE"];

/// An enum representing the available verbosity levels of the logging framework
///
/// A `LogLevel` may be compared directly to a `LogLevelFilter`.
#[repr(usize)]
#[derive(Copy, Eq, Debug)]
pub enum LogLevel {
    /// The "error" level.
    ///
    /// Designates very serious errors.
    Error = 1, // This way these line up with the discriminants for LogLevelFilter below
    /// The "warn" level.
    ///
    /// Designates hazardous situations.
    Warn,
    /// The "info" level.
    ///
    /// Designates useful information.
    Info,
    /// The "debug" level.
    ///
    /// Designates lower priority information.
    Debug,
    /// The "trace" level.
    ///
    /// Designates very low priority, often extremely verbose, information.
    Trace,
}

impl Clone for LogLevel {
    #[inline]
    fn clone(&self) -> LogLevel {
        *self
    }
}

impl PartialEq for LogLevel {
    #[inline]
    fn eq(&self, other: &LogLevel) -> bool {
        *self as usize == *other as usize
    }
}

impl PartialEq<LogLevelFilter> for LogLevel {
    #[inline]
    fn eq(&self, other: &LogLevelFilter) -> bool {
        *self as usize == *other as usize
    }
}

impl PartialOrd for LogLevel {
    #[inline]
    fn partial_cmp(&self, other: &LogLevel) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd<LogLevelFilter> for LogLevel {
    #[inline]
    fn partial_cmp(&self, other: &LogLevelFilter) -> Option<cmp::Ordering> {
        Some((*self as usize).cmp(&(*other as usize)))
    }
}

impl Ord for LogLevel {
    #[inline]
    fn cmp(&self, other: &LogLevel) -> cmp::Ordering {
        (*self as usize).cmp(&(*other as usize))
    }
}

fn ok_or<T, E>(t: Option<T>, e: E) -> Result<T, E> {
    match t {
        Some(t) => Ok(t),
        None => Err(e),
    }
}

// Reimplemented here because std::ascii is not available in libcore
fn eq_ignore_ascii_case(a: &str, b: &str) -> bool {
    fn to_ascii_uppercase(c: u8) -> u8 {
        if c >= b'a' && c <= b'z' {
            c - b'a' + b'A'
        } else {
            c
        }
    }

    if a.len() == b.len() {
        a.bytes()
         .zip(b.bytes())
         .all(|(a, b)| to_ascii_uppercase(a) == to_ascii_uppercase(b))
    } else {
        false
    }
}

impl FromStr for LogLevel {
    type Err = ();
    fn from_str(level: &str) -> Result<LogLevel, ()> {
        ok_or(LOG_LEVEL_NAMES.iter()
                    .position(|&name| eq_ignore_ascii_case(name, level))
                    .into_iter()
                    .filter(|&idx| idx != 0)
                    .map(|idx| LogLevel::from_usize(idx).unwrap())
                    .next(), ())
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.pad(LOG_LEVEL_NAMES[*self as usize])
    }
}

impl LogLevel {
    fn from_usize(u: usize) -> Option<LogLevel> {
        match u {
            1 => Some(LogLevel::Error),
            2 => Some(LogLevel::Warn),
            3 => Some(LogLevel::Info),
            4 => Some(LogLevel::Debug),
            5 => Some(LogLevel::Trace),
            _ => None
        }
    }

    /// Returns the most verbose logging level.
    #[inline]
    pub fn max() -> LogLevel {
        LogLevel::Trace
    }

    /// Converts the `LogLevel` to the equivalent `LogLevelFilter`.
    #[inline]
    pub fn to_log_level_filter(&self) -> LogLevelFilter {
        LogLevelFilter::from_usize(*self as usize).unwrap()
    }
}

/// An enum representing the available verbosity level filters of the logging
/// framework.
///
/// A `LogLevelFilter` may be compared directly to a `LogLevel`.
#[repr(usize)]
#[derive(Copy, Eq, Debug)]
pub enum LogLevelFilter {
    /// A level lower than all log levels.
    Off,
    /// Corresponds to the `Error` log level.
    Error,
    /// Corresponds to the `Warn` log level.
    Warn,
    /// Corresponds to the `Info` log level.
    Info,
    /// Corresponds to the `Debug` log level.
    Debug,
    /// Corresponds to the `Trace` log level.
    Trace,
}

// Deriving generates terrible impls of these traits

impl Clone for LogLevelFilter {
    #[inline]
    fn clone(&self) -> LogLevelFilter {
        *self
    }
}

impl PartialEq for LogLevelFilter {
    #[inline]
    fn eq(&self, other: &LogLevelFilter) -> bool {
        *self as usize == *other as usize
    }
}

impl PartialEq<LogLevel> for LogLevelFilter {
    #[inline]
    fn eq(&self, other: &LogLevel) -> bool {
        other.eq(self)
    }
}

impl PartialOrd for LogLevelFilter {
    #[inline]
    fn partial_cmp(&self, other: &LogLevelFilter) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd<LogLevel> for LogLevelFilter {
    #[inline]
    fn partial_cmp(&self, other: &LogLevel) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|x| x.reverse())
    }
}

impl Ord for LogLevelFilter {
    #[inline]
    fn cmp(&self, other: &LogLevelFilter) -> cmp::Ordering {
        (*self as usize).cmp(&(*other as usize))
    }
}

impl FromStr for LogLevelFilter {
    type Err = ();
    fn from_str(level: &str) -> Result<LogLevelFilter, ()> {
        ok_or(LOG_LEVEL_NAMES.iter()
                    .position(|&name| eq_ignore_ascii_case(name, level))
                    .map(|p| LogLevelFilter::from_usize(p).unwrap()), ())
    }
}

impl fmt::Display for LogLevelFilter {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", LOG_LEVEL_NAMES[*self as usize])
    }
}

impl LogLevelFilter {
    fn from_usize(u: usize) -> Option<LogLevelFilter> {
        match u {
            0 => Some(LogLevelFilter::Off),
            1 => Some(LogLevelFilter::Error),
            2 => Some(LogLevelFilter::Warn),
            3 => Some(LogLevelFilter::Info),
            4 => Some(LogLevelFilter::Debug),
            5 => Some(LogLevelFilter::Trace),
            _ => None
        }
    }
    /// Returns the most verbose logging level filter.
    #[inline]
    pub fn max() -> LogLevelFilter {
        LogLevelFilter::Trace
    }

    /// Converts `self` to the equivalent `LogLevel`.
    ///
    /// Returns `None` if `self` is `LogLevelFilter::Off`.
    #[inline]
    pub fn to_log_level(&self) -> Option<LogLevel> {
        LogLevel::from_usize(*self as usize)
    }
}

/// The "payload" of a log message.
pub struct LogRecord<'a> {
    metadata: LogMetadata<'a>,
    location: &'a LogLocation,
    args: fmt::Arguments<'a>,
}

impl<'a> LogRecord<'a> {
    /// The message body.
    pub fn args(&self) -> &fmt::Arguments<'a> {
        &self.args
    }

    /// Metadata about the log directive.
    pub fn metadata(&self) -> &LogMetadata {
        &self.metadata
    }

    /// The location of the log directive.
    pub fn location(&self) -> &LogLocation {
        self.location
    }

    /// The verbosity level of the message.
    pub fn level(&self) -> LogLevel {
        self.metadata.level()
    }

    /// The name of the target of the directive.
    pub fn target(&self) -> &str {
        self.metadata.target()
    }
}

/// Metadata about a log message.
pub struct LogMetadata<'a> {
    level: LogLevel,
    target: &'a str,
}

impl<'a> LogMetadata<'a> {
    /// The verbosity level of the message.
    pub fn level(&self) -> LogLevel {
        self.level
    }

    /// The name of the target of the directive.
    pub fn target(&self) -> &str {
        self.target
    }
}

/// A trait encapsulating the operations required of a logger
pub trait Log: Sync+Send {
    /// Determines if a log message with the specified metadata would be
    /// logged.
    ///
    /// This is used by the `log_enabled!` macro to allow callers to avoid
    /// expensive computation of log message arguments if the message would be
    /// discarded anyway.
    fn enabled(&self, metadata: &LogMetadata) -> bool;

    /// Logs the `LogRecord`.
    ///
    /// Note that `enabled` is *not* necessarily called before this method.
    /// Implementations of `log` should perform all necessary filtering
    /// internally.
    fn log(&self, record: &LogRecord);
}

// Just used as a dummy initial value for LOGGER
struct NopLogger;

impl Log for NopLogger {
    fn enabled(&self, _: &LogMetadata) -> bool { false }

    fn log(&self, _: &LogRecord) {}
}

/// The location of a log message.
///
/// # Warning
///
/// The fields of this struct are public so that they may be initialized by the
/// `log!` macro. They are subject to change at any time and should never be
/// accessed directly.
#[derive(Copy, Clone, Debug)]
pub struct LogLocation {
    #[doc(hidden)]
    pub __module_path: &'static str,
    #[doc(hidden)]
    pub __file: &'static str,
    #[doc(hidden)]
    pub __line: u32,
}

impl LogLocation {
    /// The module path of the message.
    pub fn module_path(&self) -> &str {
        self.__module_path
    }

    /// The source file containing the message.
    pub fn file(&self) -> &str {
        self.__file
    }

    /// The line containing the message.
    pub fn line(&self) -> u32 {
        self.__line
    }
}

/// A token providing read and write access to the global maximum log level
/// filter.
///
/// The maximum log level is used as an optimization to avoid evaluating log
/// messages that will be ignored by the logger. Any message with a level
/// higher than the maximum log level filter will be ignored. A logger should
/// make sure to keep the maximum log level filter in sync with its current
/// configuration.
#[allow(missing_copy_implementations)]
pub struct MaxLogLevelFilter(());

impl fmt::Debug for MaxLogLevelFilter {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "MaxLogLevelFilter")
    }
}

impl MaxLogLevelFilter {
    /// Gets the current maximum log level filter.
    pub fn get(&self) -> LogLevelFilter {
        max_log_level()
    }

    /// Sets the maximum log level.
    pub fn set(&self, level: LogLevelFilter) {
        MAX_LOG_LEVEL_FILTER.store(level as usize, Ordering::SeqCst)
    }
}

/// Returns the current maximum log level.
///
/// The `log!`, `error!`, `warn!`, `info!`, `debug!`, and `trace!` macros check
/// this value and discard any message logged at a higher level. The maximum
/// log level is set by the `MaxLogLevel` token passed to loggers.
#[inline(always)]
pub fn max_log_level() -> LogLevelFilter {
    unsafe { mem::transmute(MAX_LOG_LEVEL_FILTER.load(Ordering::Relaxed)) }
}

/// Sets the global logger.
///
/// The `make_logger` closure is passed a `MaxLogLevel` object, which the
/// logger should use to keep the global maximum log level in sync with the
/// highest log level that the logger will not ignore.
///
/// This function may only be called once in the lifetime of a program. Any log
/// events that occur before the call to `set_logger` completes will be
/// ignored.
///
/// This function does not typically need to be called manually. Logger
/// implementations should provide an initialization method that calls
/// `set_logger` internally.
///
/// Requires the `use_std` feature (enabled by default).
#[cfg(feature = "use_std")]
pub fn set_logger<M>(make_logger: M) -> Result<(), SetLoggerError>
        where M: FnOnce(MaxLogLevelFilter) -> Box<Log> {
    unsafe { set_logger_raw(|max_level| mem::transmute(make_logger(max_level))) }
}

/// Sets the global logger from a raw pointer.
///
/// This function is similar to `set_logger` except that it is usable in
/// `no_std` code.
///
/// The `make_logger` closure is passed a `MaxLogLevel` object, which the
/// logger should use to keep the global maximum log level in sync with the
/// highest log level that the logger will not ignore.
///
/// This function may only be called once in the lifetime of a program. Any log
/// events that occur before the call to `set_logger_raw` completes will be
/// ignored.
///
/// This function does not typically need to be called manually. Logger
/// implementations should provide an initialization method that calls
/// `set_logger_raw` internally.
///
/// # Safety
///
/// The pointer returned by `make_logger` must remain valid for the entire
/// duration of the program or until `shutdown_logger_raw` is called. In
/// addition, `shutdown_logger` *must not* be called after this function.
pub unsafe fn set_logger_raw<M>(make_logger: M) -> Result<(), SetLoggerError>
        where M: FnOnce(MaxLogLevelFilter) -> *const Log {
    if STATE.compare_and_swap(UNINITIALIZED, INITIALIZING,
                              Ordering::SeqCst) != UNINITIALIZED {
        return Err(SetLoggerError(()));
    }

    LOGGER = make_logger(MaxLogLevelFilter(()));
    STATE.store(INITIALIZED, Ordering::SeqCst);
    Ok(())
}

/// Shuts down the global logger.
///
/// This function may only be called once in the lifetime of a program, and may
/// not be called before `set_logger`. Once the global logger has been shut
/// down, it can no longer be re-initialized by `set_logger`. Any log events
/// that occur after the call to `shutdown_logger` completes will be ignored.
///
/// The logger that was originally created by the call to to `set_logger` is
/// returned on success. At that point it is guaranteed that no other threads
/// are concurrently accessing the logger object.
#[cfg(feature = "use_std")]
pub fn shutdown_logger() -> Result<Box<Log>, ShutdownLoggerError> {
    shutdown_logger_raw().map(|l| unsafe { mem::transmute(l) })
}

/// Shuts down the global logger.
///
/// This function is similar to `shutdown_logger` except that it is usable in
/// `no_std` code.
///
/// This function may only be called once in the lifetime of a program, and may
/// not be called before `set_logger_raw`. Once the global logger has been shut
/// down, it can no longer be re-initialized by `set_logger_raw`. Any log
/// events that occur after the call to `shutdown_logger_raw` completes will be
/// ignored.
///
/// The pointer that was originally passed to `set_logger_raw` is returned on
/// success. At that point it is guaranteed that no other threads are
/// concurrently accessing the logger object.
pub fn shutdown_logger_raw() -> Result<*const Log, ShutdownLoggerError> {
    // Set the global log level to stop other thread from logging
    MAX_LOG_LEVEL_FILTER.store(0, Ordering::SeqCst);

    // Set to INITIALIZING to prevent re-initialization after
    if STATE.compare_and_swap(INITIALIZED, INITIALIZING,
                              Ordering::SeqCst) != INITIALIZED {
        return Err(ShutdownLoggerError(()));
    }

    while REFCOUNT.load(Ordering::SeqCst) != 0 {
        // FIXME add a sleep here when it doesn't involve timers
    }

    unsafe {
        let logger = LOGGER;
        LOGGER = &NopLogger;
        Ok(logger)
    }
}

/// The type returned by `set_logger` if `set_logger` has already been called.
#[allow(missing_copy_implementations)]
#[derive(Debug)]
pub struct SetLoggerError(());

impl fmt::Display for SetLoggerError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "attempted to set a logger after the logging system \
                     was already initialized")
    }
}

// The Error trait is not available in libcore
#[cfg(feature = "use_std")]
impl error::Error for SetLoggerError {
    fn description(&self) -> &str { "set_logger() called multiple times" }
}

/// The type returned by `shutdown_logger_raw` if `shutdown_logger_raw` has
/// already been called or if `set_logger_raw` has not been called yet.
#[allow(missing_copy_implementations)]
#[derive(Debug)]
pub struct ShutdownLoggerError(());

impl fmt::Display for ShutdownLoggerError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "attempted to shut down the logger without an active logger")
    }
}

// The Error trait is not available in libcore
#[cfg(feature = "use_std")]
impl error::Error for ShutdownLoggerError {
    fn description(&self) -> &str { "shutdown_logger() called without an active logger" }
}

/// Registers a panic hook which logs at the error level.
///
/// The format is the same as the default panic hook . The reporting module is
/// `log::panic`.
///
/// Requires the `use_std` (enabled by default) and `nightly` features.
#[cfg(all(feature = "nightly", feature = "use_std"))]
pub fn log_panics() {
    std::panic::set_hook(Box::new(panic::log));
}

// inner module so that the reporting module is log::panic instead of log
#[cfg(all(feature = "nightly", feature = "use_std"))]
mod panic {
    use std::panic::PanicInfo;
    use std::thread;

    pub fn log(info: &PanicInfo) {
        let thread = thread::current();
        let thread = thread.name().unwrap_or("<unnamed>");

        let msg = match info.payload().downcast_ref::<&'static str>() {
            Some(s) => *s,
            None => match info.payload().downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<Any>",
            }
        };

        match info.location() {
            Some(location) => {
                error!("thread '{}' panicked at '{}': {}:{}",
                       thread,
                       msg,
                       location.file(),
                       location.line())
            }
            None => error!("thread '{}' panicked at '{}'", thread, msg),
        }
    }
}

struct LoggerGuard(&'static Log);

impl Drop for LoggerGuard {
    fn drop(&mut self) {
        REFCOUNT.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Deref for LoggerGuard {
    type Target = Log;

    fn deref(&self) -> &(Log + 'static) {
        self.0
    }
}

fn logger() -> Option<LoggerGuard> {
    REFCOUNT.fetch_add(1, Ordering::SeqCst);
    if STATE.load(Ordering::SeqCst) != INITIALIZED {
        REFCOUNT.fetch_sub(1, Ordering::SeqCst);
        None
    } else {
        Some(LoggerGuard(unsafe { &*LOGGER }))
    }
}

// WARNING
// This is not considered part of the crate's public API. It is subject to
// change at any time.
#[doc(hidden)]
pub fn __enabled(level: LogLevel, target: &str) -> bool {
    if let Some(logger) = logger() {
        logger.enabled(&LogMetadata { level: level, target: target })
    } else {
        false
    }
}

// WARNING
// This is not considered part of the crate's public API. It is subject to
// change at any time.
#[doc(hidden)]
pub fn __log(level: LogLevel, target: &str, loc: &LogLocation,
             args: fmt::Arguments) {
    if let Some(logger) = logger() {
        let record = LogRecord {
            metadata: LogMetadata {
                level: level,
                target: target,
            },
            location: loc,
            args: args
        };
        logger.log(&record)
    }
}

// WARNING
// This is not considered part of the crate's public API. It is subject to
// change at any time.
#[inline(always)]
#[doc(hidden)]
pub fn __static_max_level() -> LogLevelFilter {
    if !cfg!(debug_assertions) {
        // This is a release build. Check `release_max_level_*` first.
        if cfg!(feature = "release_max_level_off") {
            return LogLevelFilter::Off
        } else if cfg!(feature = "release_max_level_error") {
            return LogLevelFilter::Error
        } else if cfg!(feature = "release_max_level_warn") {
            return LogLevelFilter::Warn
        } else if cfg!(feature = "release_max_level_info") {
            return LogLevelFilter::Info
        } else if cfg!(feature = "release_max_level_debug") {
            return LogLevelFilter::Debug
        } else if cfg!(feature = "release_max_level_trace") {
            return LogLevelFilter::Trace
        }
    }
    if cfg!(feature = "max_level_off") {
        LogLevelFilter::Off
    } else if cfg!(feature = "max_level_error") {
        LogLevelFilter::Error
    } else if cfg!(feature = "max_level_warn") {
        LogLevelFilter::Warn
    } else if cfg!(feature = "max_level_info") {
        LogLevelFilter::Info
    } else if cfg!(feature = "max_level_debug") {
        LogLevelFilter::Debug
    } else {
        LogLevelFilter::Trace
    }
}

#[cfg(test)]
mod tests {
     extern crate std;
     use tests::std::string::ToString;
     use super::{LogLevel, LogLevelFilter};

     #[test]
     fn test_loglevelfilter_from_str() {
         let tests = [
             ("off",   Ok(LogLevelFilter::Off)),
             ("error", Ok(LogLevelFilter::Error)),
             ("warn",  Ok(LogLevelFilter::Warn)),
             ("info",  Ok(LogLevelFilter::Info)),
             ("debug", Ok(LogLevelFilter::Debug)),
             ("trace", Ok(LogLevelFilter::Trace)),
             ("OFF",   Ok(LogLevelFilter::Off)),
             ("ERROR", Ok(LogLevelFilter::Error)),
             ("WARN",  Ok(LogLevelFilter::Warn)),
             ("INFO",  Ok(LogLevelFilter::Info)),
             ("DEBUG", Ok(LogLevelFilter::Debug)),
             ("TRACE", Ok(LogLevelFilter::Trace)),
             ("asdf",  Err(())),
         ];
         for &(s, ref expected) in &tests {
             assert_eq!(expected, &s.parse());
         }
     }

     #[test]
     fn test_loglevel_from_str() {
         let tests = [
             ("OFF",   Err(())),
             ("error", Ok(LogLevel::Error)),
             ("warn",  Ok(LogLevel::Warn)),
             ("info",  Ok(LogLevel::Info)),
             ("debug", Ok(LogLevel::Debug)),
             ("trace", Ok(LogLevel::Trace)),
             ("ERROR", Ok(LogLevel::Error)),
             ("WARN",  Ok(LogLevel::Warn)),
             ("INFO",  Ok(LogLevel::Info)),
             ("DEBUG", Ok(LogLevel::Debug)),
             ("TRACE", Ok(LogLevel::Trace)),
             ("asdf",  Err(())),
         ];
         for &(s, ref expected) in &tests {
             assert_eq!(expected, &s.parse());
         }
     }

     #[test]
     fn test_loglevel_show() {
         assert_eq!("INFO", LogLevel::Info.to_string());
         assert_eq!("ERROR", LogLevel::Error.to_string());
     }

     #[test]
     fn test_loglevelfilter_show() {
         assert_eq!("OFF", LogLevelFilter::Off.to_string());
         assert_eq!("ERROR", LogLevelFilter::Error.to_string());
     }

     #[test]
     fn test_cross_cmp() {
         assert!(LogLevel::Debug > LogLevelFilter::Error);
         assert!(LogLevelFilter::Warn < LogLevel::Trace);
         assert!(LogLevelFilter::Off < LogLevel::Error);
     }

     #[test]
     fn test_cross_eq() {
         assert!(LogLevel::Error == LogLevelFilter::Error);
         assert!(LogLevelFilter::Off != LogLevel::Error);
         assert!(LogLevel::Trace == LogLevelFilter::Trace);
     }

     #[test]
     fn test_to_log_level() {
         assert_eq!(Some(LogLevel::Error), LogLevelFilter::Error.to_log_level());
         assert_eq!(None, LogLevelFilter::Off.to_log_level());
         assert_eq!(Some(LogLevel::Debug), LogLevelFilter::Debug.to_log_level());
     }

     #[test]
     fn test_to_log_level_filter() {
         assert_eq!(LogLevelFilter::Error, LogLevel::Error.to_log_level_filter());
         assert_eq!(LogLevelFilter::Trace, LogLevel::Trace.to_log_level_filter());
     }

     #[test]
     #[cfg(feature = "use_std")]
     fn test_error_trait() {
         use std::error::Error;
         use super::SetLoggerError;
         let e = SetLoggerError(());
         assert_eq!(e.description(), "set_logger() called multiple times");
     }
}
