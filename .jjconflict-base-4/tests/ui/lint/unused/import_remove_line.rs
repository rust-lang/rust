//@ run-rustfix
//@ check-pass

#![crate_type = "lib"]
#![warn(unused_imports)]

use std::time::{Duration, Instant};
//~^ WARN unused imports
use std::time::SystemTime;
//~^ WARN unused import
use std::time::SystemTimeError;use std::time::TryFromFloatSecsError;
//~^ WARN unused import
//~| WARN unused import
