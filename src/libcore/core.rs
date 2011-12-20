// Top-level, visible-everywhere definitions.

// Export type option as a synonym for option::t and export the some and none
// tag constructors.

import option::{some,  none};
import option = option::t;
export option, some, none;

// Export the log levels as global constants. Higher levels mean
// more-verbosity. Error is the bottom level, default logging level is
// warn-and-below.

const error : int = 0;
const warn : int = 1;
const info : int = 2;
const debug : int = 3;
