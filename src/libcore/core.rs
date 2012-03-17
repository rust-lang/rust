// Top-level, visible-everywhere definitions.

// Export various ubiquitous types, constructors, methods.

import option::{some, none};
import option = option::option;
import path = path::path;
import vec::vec_len;
import str::extensions;
import option::extensions;
export path, option, some, none, vec_len, unreachable;
export extensions;

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

#[doc = "
A standard function to use to indicate unreachable code. Because the
function is guaranteed to fail typestate will correctly identify
any code paths following the appearance of this function as unreachable.
"]
fn unreachable() -> ! {
    fail "Internal error: entered unreachable code";
}

