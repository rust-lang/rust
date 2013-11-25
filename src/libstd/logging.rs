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

Logging

This module is used by the compiler when emitting output for the logging family
of macros. The methods of this module shouldn't necessarily be used directly,
but rather through the logging macros defined.

There are five macros that the logging subsystem uses:

* `log!(level, ...)` - the generic logging macro, takes a level as a u32 and any
                       related `format!` arguments
* `debug!(...)` - a macro hard-wired to the log level of 4
* `info!(...)` - a macro hard-wired to the log level of 3
* `warn!(...)` - a macro hard-wired to the log level of 2
* `error!(...)` - a macro hard-wired to the log level of 1

All of these macros use the same style of syntax as the `format!` syntax
extension. Details about the syntax can be found in the documentation of
`std::fmt` along with the Rust tutorial/manual

## Enabling logging

Log levels are controlled on a per-module basis, and by default all logging is
disabled except for `error!` (a log level of 1). Logging is controlled via the
`RUST_LOG` environment variable. The value of this environment variable is a
comma-separated list of logging directives. A logging directive is of the form:

```
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

```
hello                // turns on all logging for the 'hello' module
info                 // turns on all info logging
hello=debug          // turns on debug logging for 'hello'
hello=3              // turns on info logging for 'hello'
hello,std::hashmap   // turns on hello, and std's hashmap logging
error,hello=warn     // turn on global error logging and also warn for hello
```

## Performance and Side Effects

Each of these macros will expand to code similar to:

```rust
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
use option::*;
use rt::local::Local;
use rt::logging::{Logger, StdErrLogger};
use rt::task::Task;

/// This function is called directly by the compiler when using the logging
/// macros. This function does not take into account whether the log level
/// specified is active or not, it will always log something if this method is
/// called.
///
/// It is not recommended to call this function directly, rather it should be
/// invoked through the logging family of macros.
pub fn log(_level: u32, args: &fmt::Arguments) {
    unsafe {
        let optional_task: Option<*mut Task> = Local::try_unsafe_borrow();
        match optional_task {
            Some(local) => {
                match (*local).logger {
                    // Use the available logger if we have one
                    Some(ref mut logger) => return logger.log(args),
                    None => {
                        let mut logger = StdErrLogger::new();
                        logger.log(args);
                        (*local).logger = Some(logger);
                    }
                }
            }
            None => {}
        }
        // There is no logger anywhere, just write to stderr
        let mut logger = StdErrLogger::new();
        logger.log(args);
    }
}
