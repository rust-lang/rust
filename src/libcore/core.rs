// Top-level, visible-everywhere definitions.

// Export type option as a synonym for option and export the some and none
// enum constructors.

import option::{some,  none};
import option = option::t;
export option, some, none;

// Export the log levels as global constants. Higher levels mean
// more-verbosity. Error is the bottom level, default logging level is
// warn-and-below.

export error, warn, info, debug;

#[doc = "The error log level"]
const error : u32 = 0_u32;
#[doc = "The warning log level"]
const warn : u32 = 1_u32;
#[doc = "The info log level"]
const info : u32 = 2_u32;
#[doc = "The debug log level"]
const debug : u32 = 3_u32;

// A curious inner-module that's not exported that contains the binding
// 'core' so that macro-expanded references to core::error and such
// can be resolved within libcore.
mod core {
    const error : u32 = 0_u32;
    const warn : u32 = 1_u32;
    const info : u32 = 2_u32;
    const debug : u32 = 3_u32;
}

// Similar to above. Some magic to make core testable.
#[cfg(test)]
mod std {
    use std;
    import std::test;
}
