// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

This module is used by the compiler when emitting output for the logging family
of macros. The methods of this module shouldn't necessarily be used directly,
but rather through the logging macros defined.

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

```ignore
path::to::module=log_level
```

The path to the module is rooted in the name of the crate it was compiled for,
so if your program is contained in a file `hello.rs`, for example, to turn on
logging for this file you would use a value of `RUST_LOG=hello`. Furthermore,
this path is a prefix-search, so all modules nested in the specified module will
also have logging enabled.

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

```ignore
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
    ::std::logging::log(log_level, format!(...));
}
```

What this means is that each of these macros are very cheap at runtime if
they're turned off (just a load and an integer comparison). This also means that
if logging is disabled, none of the components of the log will be executed.

## Useful Values

For convenience, if a value of `::help` is set for `RUST_LOG`, a program will
start, print out all modules registered for logging, and then exit.

*/

use fmt;
use io::LineBufferedWriter;
use io;
use io::Writer;
use mem::replace;
use ops::Drop;
use option::{Some, None, Option};
use prelude::drop;
use result::{Ok, Err};
use rt::local::Local;
use rt::task::Task;

/// Debug log level
pub static DEBUG: u32 = 4;
/// Info log level
pub static INFO: u32 = 3;
/// Warn log level
pub static WARN: u32 = 2;
/// Error log level
pub static ERROR: u32 = 1;

/// A trait used to represent an interface to a task-local logger. Each task
/// can have its own custom logger which can respond to logging messages
/// however it likes.
pub trait Logger {
    /// Logs a single message described by the `args` structure. The level is
    /// provided in case you want to do things like color the message, etc.
    fn log(&mut self, level: u32, args: &fmt::Arguments);
}

struct DefaultLogger {
    handle: LineBufferedWriter<io::stdio::StdWriter>,
}

impl Logger for DefaultLogger {
    // by default, just ignore the level
    fn log(&mut self, _level: u32, args: &fmt::Arguments) {
        match fmt::writeln(&mut self.handle, args) {
            Err(e) => fail!("failed to log: {}", e),
            Ok(()) => {}
        }
    }
}

impl Drop for DefaultLogger {
    fn drop(&mut self) {
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
pub fn log(level: u32, args: &fmt::Arguments) {
    // See io::stdio::with_task_stdout for why there's a few dances here. The
    // gist of it is that arbitrary code can run during logging (and set an
    // arbitrary logging handle into the task) so we need to be careful that the
    // local task is in TLS while we're running arbitrary code.
    let mut logger = {
        let mut task = Local::borrow(None::<Task>);
        task.get().logger.take()
    };

    if logger.is_none() {
        logger = Some(~DefaultLogger {
            handle: LineBufferedWriter::new(io::stderr()),
        } as ~Logger);
    }
    logger.get_mut_ref().log(level, args);

    let mut task = Local::borrow(None::<Task>);
    let prev = replace(&mut task.get().logger, logger);
    drop(task);
    drop(prev);
}

/// Replaces the task-local logger with the specified logger, returning the old
/// logger.
pub fn set_logger(logger: ~Logger) -> Option<~Logger> {
    let mut task = Local::borrow(None::<Task>);
    replace(&mut task.get().logger, Some(logger))
}
