// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
/// The standard logging macro.
///
/// This macro will generically log with the specified `LogLevel` and `format!`
/// based argument list.
///
/// The `max_level_*` features can be used to statically disable logging at
/// various levels.
#[macro_export]
macro_rules! log {
    (target: $target:expr, $lvl:expr, $($arg:tt)+) => ({
        static _LOC: $crate::LogLocation = $crate::LogLocation {
            __line: line!(),
            __file: file!(),
            __module_path: module_path!(),
        };
        let lvl = $lvl;
        if lvl <= $crate::__static_max_level() && lvl <= $crate::max_log_level() {
            $crate::__log(lvl, $target, &_LOC, format_args!($($arg)+))
        }
    });
    ($lvl:expr, $($arg:tt)+) => (log!(target: module_path!(), $lvl, $($arg)+))
}

/// Logs a message at the error level.
///
/// Logging at this level is disabled if the `max_level_off` feature is present.
#[macro_export]
macro_rules! error {
    (target: $target:expr, $($arg:tt)*) => (
        log!(target: $target, $crate::LogLevel::Error, $($arg)*);
    );
    ($($arg:tt)*) => (
        log!($crate::LogLevel::Error, $($arg)*);
    )
}

/// Logs a message at the warn level.
///
/// Logging at this level is disabled if any of the following features are
/// present: `max_level_off` or `max_level_error`.
///
/// When building in release mode (i.e., without the `debug_assertions` option),
/// logging at this level is also disabled if any of the following features are
/// present: `release_max_level_off` or `max_level_error`.
#[macro_export]
macro_rules! warn {
    (target: $target:expr, $($arg:tt)*) => (
        log!(target: $target, $crate::LogLevel::Warn, $($arg)*);
    );
    ($($arg:tt)*) => (
        log!($crate::LogLevel::Warn, $($arg)*);
    )
}

/// Logs a message at the info level.
///
/// Logging at this level is disabled if any of the following features are
/// present: `max_level_off`, `max_level_error`, or `max_level_warn`.
///
/// When building in release mode (i.e., without the `debug_assertions` option),
/// logging at this level is also disabled if any of the following features are
/// present: `release_max_level_off`, `release_max_level_error`, or
/// `release_max_level_warn`.
#[macro_export]
macro_rules! info {
    (target: $target:expr, $($arg:tt)*) => (
        log!(target: $target, $crate::LogLevel::Info, $($arg)*);
    );
    ($($arg:tt)*) => (
        log!($crate::LogLevel::Info, $($arg)*);
    )
}

/// Logs a message at the debug level.
///
/// Logging at this level is disabled if any of the following features are
/// present: `max_level_off`, `max_level_error`, `max_level_warn`, or
/// `max_level_info`.
///
/// When building in release mode (i.e., without the `debug_assertions` option),
/// logging at this level is also disabled if any of the following features are
/// present: `release_max_level_off`, `release_max_level_error`,
/// `release_max_level_warn`, or `release_max_level_info`.
#[macro_export]
macro_rules! debug {
    (target: $target:expr, $($arg:tt)*) => (
        log!(target: $target, $crate::LogLevel::Debug, $($arg)*);
    );
    ($($arg:tt)*) => (
        log!($crate::LogLevel::Debug, $($arg)*);
    )
}

/// Logs a message at the trace level.
///
/// Logging at this level is disabled if any of the following features are
/// present: `max_level_off`, `max_level_error`, `max_level_warn`,
/// `max_level_info`, or `max_level_debug`.
///
/// When building in release mode (i.e., without the `debug_assertions` option),
/// logging at this level is also disabled if any of the following features are
/// present: `release_max_level_off`, `release_max_level_error`,
/// `release_max_level_warn`, `release_max_level_info`, or
/// `release_max_level_debug`.
#[macro_export]
macro_rules! trace {
    (target: $target:expr, $($arg:tt)*) => (
        log!(target: $target, $crate::LogLevel::Trace, $($arg)*);
    );
    ($($arg:tt)*) => (
        log!($crate::LogLevel::Trace, $($arg)*);
    )
}

/// Determines if a message logged at the specified level in that module will
/// be logged.
///
/// This can be used to avoid expensive computation of log message arguments if
/// the message would be ignored anyway.
///
/// # Examples
///
/// ```rust
/// # #[macro_use]
/// # extern crate log;
/// use log::LogLevel::Debug;
///
/// # fn foo() {
/// if log_enabled!(Debug) {
///     let data = expensive_call();
///     debug!("expensive debug data: {} {}", data.x, data.y);
/// }
/// # }
/// # struct Data { x: u32, y: u32 }
/// # fn expensive_call() -> Data { Data { x: 0, y: 0 } }
/// # fn main() {}
/// ```
#[macro_export]
macro_rules! log_enabled {
    (target: $target:expr, $lvl:expr) => ({
        let lvl = $lvl;
        lvl <= $crate::__static_max_level() && lvl <= $crate::max_log_level() &&
            $crate::__enabled(lvl, $target)
    });
    ($lvl:expr) => (log_enabled!(target: module_path!(), $lvl))
}
