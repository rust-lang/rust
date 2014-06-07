// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Utilities for program-wide and customizable logging

## Example

```
#![feature(phase)]
#[phase(plugin, link)] extern crate log;

fn main() {
    debug!("this is a debug {}", "message");
    error!("this is printed by default");

    if log_enabled!(log::INFO) {
        let x = 3 * 4; // expensive computation
        info!("the answer was: {}", x);
    }
}
```

## Logging Macros

There are five macros that the logging subsystem uses:

* `log!(level, ...)` - the generic logging macro, takes a level as a u32 and any
                       related `format!` arguments
* `debug!(...)` - a macro hard-wired to the log level of `DEBUG`
* `info!(...)` - a macro hard-wired to the log level of `INFO`
* `warn!(...)` - a macro hard-wired to the log level of `WARN`
* `error!(...)` - a macro hard-wired to the log level of `ERROR`

All of these macros use the same style of syntax as the `format!` syntax
extension. Details about the syntax can be found in the documentation of
`std::fmt` along with the Rust tutorial/manual.

If you want to check at runtime if a given logging level is enabled (e.g. if the
information you would want to log is expensive to produce), you can use the
following macro:

* `log_enabled!(level)` - returns true if logging of the given level is enabled

## Enabling logging

Log levels are controlled on a per-module basis, and by default all logging is
disabled except for `error!` (a log level of 1). Logging is controlled via the
`RUST_LOG` environment variable. The value of this environment variable is a
comma-separated list of logging directives. A logging directive is of the form:

```text
path::to::module=log_level
```

The path to the module is rooted in the name of the crate it was compiled for,
so if your program is contained in a file `hello.rs`, for example, to turn on
logging for this file you would use a value of `RUST_LOG=hello`.
Furthermore, this path is a prefix-search, so all modules nested in the
specified module will also have logging enabled.

The actual `log_level` is optional to specify. If omitted, all logging will be
enabled. If specified, the it must be either a numeric in the range of 1-255, or
it must be one of the strings `debug`, `error`, `info`, or `warn`. If a numeric
is specified, then all logging less than or equal to that numeral is enabled.
For example, if logging level 3 is active, error, warn, and info logs will be
printed, but debug will be omitted.

As the log level for a module is optional, the module to enable logging for is
also optional. If only a `log_level` is provided, then the global log level for
all modules is set to this value.

Some examples of valid values of `RUST_LOG` are:

```text
hello                // turns on all logging for the 'hello' module
info                 // turns on all info logging
hello=debug          // turns on debug logging for 'hello'
hello=3              // turns on info logging for 'hello'
hello,std::option    // turns on hello, and std's option logging
error,hello=warn     // turn on global error logging and also warn for hello
```

## Performance and Side Effects

Each of these macros will expand to code similar to:

```rust,ignore
if log_level <= my_module_log_level() {
    ::log::log(log_level, format!(...));
}
```

What this means is that each of these macros are very cheap at runtime if
they're turned off (just a load and an integer comparison). This also means that
if logging is disabled, none of the components of the log will be executed.

*/

#![crate_id = "log#0.11.0-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(macro_rules)]
#![deny(missing_doc, deprecated_owned_vector)]

use std::fmt;
use std::io::LineBufferedWriter;
use std::io;
use std::mem;
use std::os;
use std::rt;
use std::slice;
use std::sync::{Once, ONCE_INIT};

use directive::LOG_LEVEL_NAMES;

pub mod macros;
mod directive;

/// Maximum logging level of a module that can be specified. Common logging
/// levels are found in the DEBUG/INFO/WARN/ERROR constants.
pub static MAX_LOG_LEVEL: u32 = 255;

/// The default logging level of a crate if no other is specified.
static DEFAULT_LOG_LEVEL: u32 = 1;

/// An unsafe constant that is the maximum logging level of any module
/// specified. This is the first line of defense to determining whether a
/// logging statement should be run.
static mut LOG_LEVEL: u32 = MAX_LOG_LEVEL;

static mut DIRECTIVES: *Vec<directive::LogDirective> =
    0 as *Vec<directive::LogDirective>;

/// Debug log level
pub static DEBUG: u32 = 4;
/// Info log level
pub static INFO: u32 = 3;
/// Warn log level
pub static WARN: u32 = 2;
/// Error log level
pub static ERROR: u32 = 1;

local_data_key!(local_logger: Box<Logger:Send>)

/// A trait used to represent an interface to a task-local logger. Each task
/// can have its own custom logger which can respond to logging messages
/// however it likes.
pub trait Logger {
    /// Logs a single message described by the `record`.
    fn log(&mut self, record: &LogRecord);
}

struct DefaultLogger {
    handle: LineBufferedWriter<io::stdio::StdWriter>,
}

/// Wraps the log level with fmt implementations.
#[deriving(PartialEq, PartialOrd)]
pub struct LogLevel(pub u32);

impl fmt::Show for LogLevel {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let LogLevel(level) = *self;
        match LOG_LEVEL_NAMES.get(level as uint - 1) {
            Some(name) => name.fmt(fmt),
            None => level.fmt(fmt)
        }
    }
}

impl fmt::Signed for LogLevel {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let LogLevel(level) = *self;
        write!(fmt, "{}", level)
    }
}

impl Logger for DefaultLogger {
    fn log(&mut self, record: &LogRecord) {
        match writeln!(&mut self.handle,
                       "{}:{}: {}",
                       record.level,
                       record.module_path,
                       record.args) {
            Err(e) => fail!("failed to log: {}", e),
            Ok(()) => {}
        }
    }
}

impl Drop for DefaultLogger {
    fn drop(&mut self) {
        // FIXME(#12628): is failure the right thing to do?
        match self.handle.flush() {
            Err(e) => fail!("failed to flush a logger: {}", e),
            Ok(()) => {}
        }
    }
}

/// This function is called directly by the compiler when using the logging
/// macros. This function does not take into account whether the log level
/// specified is active or not, it will always log something if this method is
/// called.
///
/// It is not recommended to call this function directly, rather it should be
/// invoked through the logging family of macros.
#[doc(hidden)]
pub fn log(level: u32, loc: &'static LogLocation, args: &fmt::Arguments) {
    // Completely remove the local logger from TLS in case anyone attempts to
    // frob the slot while we're doing the logging. This will destroy any logger
    // set during logging.
    let mut logger = local_logger.replace(None).unwrap_or_else(|| {
        box DefaultLogger { handle: io::stderr() } as Box<Logger:Send>
    });
    logger.log(&LogRecord {
        level: LogLevel(level),
        args: args,
        file: loc.file,
        module_path: loc.module_path,
        line: loc.line,
    });
    local_logger.replace(Some(logger));
}

/// Getter for the global log level. This is a function so that it can be called
/// safely
#[doc(hidden)]
#[inline(always)]
pub fn log_level() -> u32 { unsafe { LOG_LEVEL } }

/// Replaces the task-local logger with the specified logger, returning the old
/// logger.
pub fn set_logger(logger: Box<Logger:Send>) -> Option<Box<Logger:Send>> {
    local_logger.replace(Some(logger))
}

/// A LogRecord is created by the logging macros, and passed as the only
/// argument to Loggers.
#[deriving(Show)]
pub struct LogRecord<'a> {

    /// The module path of where the LogRecord originated.
    pub module_path: &'a str,

    /// The LogLevel of this record.
    pub level: LogLevel,

    /// The arguments from the log line.
    pub args: &'a fmt::Arguments<'a>,

    /// The file of where the LogRecord originated.
    pub file: &'a str,

    /// The line number of where the LogRecord originated.
    pub line: uint,
}

#[doc(hidden)]
pub struct LogLocation {
    pub module_path: &'static str,
    pub file: &'static str,
    pub line: uint,
}

/// Tests whether a given module's name is enabled for a particular level of
/// logging. This is the second layer of defense about determining whether a
/// module's log statement should be emitted or not.
#[doc(hidden)]
pub fn mod_enabled(level: u32, module: &str) -> bool {
    static mut INIT: Once = ONCE_INIT;
    unsafe { INIT.doit(init); }

    // It's possible for many threads are in this function, only one of them
    // will peform the global initialization, but all of them will need to check
    // again to whether they should really be here or not. Hence, despite this
    // check being expanded manually in the logging macro, this function checks
    // the log level again.
    if level > unsafe { LOG_LEVEL } { return false }

    // This assertion should never get tripped unless we're in an at_exit
    // handler after logging has been torn down and a logging attempt was made.
    assert!(unsafe { !DIRECTIVES.is_null() });

    enabled(level, module, unsafe { (*DIRECTIVES).iter() })
}

fn enabled(level: u32,
           module: &str,
           iter: slice::Items<directive::LogDirective>)
           -> bool {
    // Search for the longest match, the vector is assumed to be pre-sorted.
    for directive in iter.rev() {
        match directive.name {
            Some(ref name) if !module.starts_with(name.as_slice()) => {},
            Some(..) | None => {
                return level <= directive.level
            }
        }
    }
    level <= DEFAULT_LOG_LEVEL
}

/// Initialize logging for the current process.
///
/// This is not threadsafe at all, so initialization os performed through a
/// `Once` primitive (and this function is called from that primitive).
fn init() {
    let mut directives = match os::getenv("RUST_LOG") {
        Some(spec) => directive::parse_logging_spec(spec.as_slice()),
        None => Vec::new(),
    };

    // Sort the provided directives by length of their name, this allows a
    // little more efficient lookup at runtime.
    directives.sort_by(|a, b| {
        let alen = a.name.as_ref().map(|a| a.len()).unwrap_or(0);
        let blen = b.name.as_ref().map(|b| b.len()).unwrap_or(0);
        alen.cmp(&blen)
    });

    let max_level = {
        let max = directives.iter().max_by(|d| d.level);
        max.map(|d| d.level).unwrap_or(DEFAULT_LOG_LEVEL)
    };

    unsafe {
        LOG_LEVEL = max_level;

        assert!(DIRECTIVES.is_null());
        DIRECTIVES = mem::transmute(box directives);

        // Schedule the cleanup for this global for when the runtime exits.
        rt::at_exit(proc() {
            assert!(!DIRECTIVES.is_null());
            let _directives: Box<Vec<directive::LogDirective>> =
                mem::transmute(DIRECTIVES);
            DIRECTIVES = 0 as *Vec<directive::LogDirective>;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::enabled;
    use directive::LogDirective;

    #[test]
    fn match_full_path() {
        let dirs = [
            LogDirective {
                name: Some("crate2".to_string()),
                level: 3
            },
            LogDirective {
                name: Some("crate1::mod1".to_string()),
                level: 2
            }
        ];
        assert!(enabled(2, "crate1::mod1", dirs.iter()));
        assert!(!enabled(3, "crate1::mod1", dirs.iter()));
        assert!(enabled(3, "crate2", dirs.iter()));
        assert!(!enabled(4, "crate2", dirs.iter()));
    }

    #[test]
    fn no_match() {
        let dirs = [
            LogDirective { name: Some("crate2".to_string()), level: 3 },
            LogDirective { name: Some("crate1::mod1".to_string()), level: 2 }
        ];
        assert!(!enabled(2, "crate3", dirs.iter()));
    }

    #[test]
    fn match_beginning() {
        let dirs = [
            LogDirective { name: Some("crate2".to_string()), level: 3 },
            LogDirective { name: Some("crate1::mod1".to_string()), level: 2 }
        ];
        assert!(enabled(3, "crate2::mod1", dirs.iter()));
    }

    #[test]
    fn match_beginning_longest_match() {
        let dirs = [
            LogDirective { name: Some("crate2".to_string()), level: 3 },
            LogDirective { name: Some("crate2::mod".to_string()), level: 4 },
            LogDirective { name: Some("crate1::mod1".to_string()), level: 2 }
        ];
        assert!(enabled(4, "crate2::mod1", dirs.iter()));
        assert!(!enabled(4, "crate2", dirs.iter()));
    }

    #[test]
    fn match_default() {
        let dirs = [
            LogDirective { name: None, level: 3 },
            LogDirective { name: Some("crate1::mod1".to_string()), level: 2 }
        ];
        assert!(enabled(2, "crate1::mod1", dirs.iter()));
        assert!(enabled(3, "crate2::mod2", dirs.iter()));
    }

    #[test]
    fn zero_level() {
        let dirs = [
            LogDirective { name: None, level: 3 },
            LogDirective { name: Some("crate1::mod1".to_string()), level: 0 }
        ];
        assert!(!enabled(1, "crate1::mod1", dirs.iter()));
        assert!(enabled(3, "crate2::mod2", dirs.iter()));
    }
}
