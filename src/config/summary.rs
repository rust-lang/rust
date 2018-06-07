// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::default::Default;
use std::time::{Duration, Instant};

/// A summary of a Rustfmt run.
#[derive(Debug, Default, Clone, Copy)]
pub struct Summary {
    // Encountered e.g. an IO error.
    has_operational_errors: bool,

    // Failed to reformat code because of parsing errors.
    has_parsing_errors: bool,

    // Code is valid, but it is impossible to format it properly.
    has_formatting_errors: bool,

    // Code contains macro call that was unable to format.
    pub(crate) has_macro_format_failure: bool,

    // Failed a check, such as the license check or other opt-in checking.
    has_check_errors: bool,

    /// Formatted code differs from existing code (--check only).
    pub has_diff: bool,

    // Keeps track of time spent in parsing and formatting steps.
    timer: Timer,
}

impl Summary {
    pub(crate) fn mark_parse_time(&mut self) {
        self.timer = self.timer.done_parsing();
    }

    pub(crate) fn mark_format_time(&mut self) {
        self.timer = self.timer.done_formatting();
    }

    /// Returns the time it took to parse the source files in nanoseconds.
    pub(crate) fn get_parse_time(&self) -> Option<Duration> {
        match self.timer {
            Timer::DoneParsing(init, parse_time) | Timer::DoneFormatting(init, parse_time, _) => {
                // This should never underflow since `Instant::now()` guarantees monotonicity.
                Some(parse_time.duration_since(init))
            }
            Timer::Initialized(..) => None,
        }
    }

    /// Returns the time it took to go from the parsed AST to the formatted output. Parsing time is
    /// not included.
    pub(crate) fn get_format_time(&self) -> Option<Duration> {
        match self.timer {
            Timer::DoneFormatting(_init, parse_time, format_time) => {
                Some(format_time.duration_since(parse_time))
            }
            Timer::DoneParsing(..) | Timer::Initialized(..) => None,
        }
    }

    pub fn has_operational_errors(&self) -> bool {
        self.has_operational_errors
    }

    pub fn has_parsing_errors(&self) -> bool {
        self.has_parsing_errors
    }

    pub fn has_formatting_errors(&self) -> bool {
        self.has_formatting_errors
    }

    pub fn has_check_errors(&self) -> bool {
        self.has_check_errors
    }

    pub(crate) fn has_macro_formatting_failure(&self) -> bool {
        self.has_macro_format_failure
    }

    pub fn add_operational_error(&mut self) {
        self.has_operational_errors = true;
    }

    pub(crate) fn add_parsing_error(&mut self) {
        self.has_parsing_errors = true;
    }

    pub(crate) fn add_formatting_error(&mut self) {
        self.has_formatting_errors = true;
    }

    pub(crate) fn add_check_error(&mut self) {
        self.has_check_errors = true;
    }

    pub(crate) fn add_diff(&mut self) {
        self.has_diff = true;
    }

    pub(crate) fn add_macro_foramt_failure(&mut self) {
        self.has_macro_format_failure = true;
    }

    pub fn has_no_errors(&self) -> bool {
        !(self.has_operational_errors
            || self.has_parsing_errors
            || self.has_formatting_errors
            || self.has_diff)
    }

    /// Combine two summaries together.
    pub fn add(&mut self, other: Summary) {
        self.has_operational_errors |= other.has_operational_errors;
        self.has_formatting_errors |= other.has_formatting_errors;
        self.has_parsing_errors |= other.has_parsing_errors;
        self.has_check_errors |= other.has_check_errors;
        self.has_diff |= other.has_diff;
    }
}

#[derive(Clone, Copy, Debug)]
enum Timer {
    Initialized(Instant),
    DoneParsing(Instant, Instant),
    DoneFormatting(Instant, Instant, Instant),
}

impl Default for Timer {
    fn default() -> Self {
        Timer::Initialized(Instant::now())
    }
}

impl Timer {
    fn done_parsing(self) -> Self {
        match self {
            Timer::Initialized(init_time) => Timer::DoneParsing(init_time, Instant::now()),
            _ => panic!("Timer can only transition to DoneParsing from Initialized state"),
        }
    }

    fn done_formatting(self) -> Self {
        match self {
            Timer::DoneParsing(init_time, parse_time) => {
                Timer::DoneFormatting(init_time, parse_time, Instant::now())
            }
            _ => panic!("Timer can only transition to DoneFormatting from DoneParsing state"),
        }
    }
}
