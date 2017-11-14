use time::{precise_time_ns, Duration};
use std::default::Default;

#[must_use]
#[derive(Debug, Default, Clone, Copy)]
pub struct Summary {
    // Encountered e.g. an IO error.
    has_operational_errors: bool,

    // Failed to reformat code because of parsing errors.
    has_parsing_errors: bool,

    // Code is valid, but it is impossible to format it properly.
    has_formatting_errors: bool,

    // Formatted code differs from existing code (write-mode diff only).
    pub has_diff: bool,

    // Keeps track of time spent in parsing and formatting steps.
    timer: Timer,
}

impl Summary {
    pub fn mark_parse_time(&mut self) {
        self.timer = self.timer.done_parsing();
    }

    pub fn mark_format_time(&mut self) {
        self.timer = self.timer.done_formatting();
    }

    /// Returns the time it took to parse the source files in nanoseconds.
    pub fn get_parse_time(&self) -> Option<Duration> {
        match self.timer {
            Timer::DoneParsing(init, parse_time) | Timer::DoneFormatting(init, parse_time, _) => {
                // This should never underflow since `precise_time_ns()` guarantees monotonicity.
                Some(Duration::nanoseconds((parse_time - init) as i64))
            }
            Timer::Initialized(..) => None,
        }
    }

    /// Returns the time it took to go from the parsed AST to the formatted output. Parsing time is
    /// not included.
    pub fn get_format_time(&self) -> Option<Duration> {
        match self.timer {
            Timer::DoneFormatting(_init, parse_time, format_time) => {
                Some(Duration::nanoseconds((format_time - parse_time) as i64))
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

    pub fn add_operational_error(&mut self) {
        self.has_operational_errors = true;
    }

    pub fn add_parsing_error(&mut self) {
        self.has_parsing_errors = true;
    }

    pub fn add_formatting_error(&mut self) {
        self.has_formatting_errors = true;
    }

    pub fn add_diff(&mut self) {
        self.has_diff = true;
    }

    pub fn has_no_errors(&self) -> bool {
        !(self.has_operational_errors || self.has_parsing_errors || self.has_formatting_errors
            || self.has_diff)
    }

    pub fn add(&mut self, other: Summary) {
        self.has_operational_errors |= other.has_operational_errors;
        self.has_formatting_errors |= other.has_formatting_errors;
        self.has_parsing_errors |= other.has_parsing_errors;
        self.has_diff |= other.has_diff;
    }

    pub fn print_exit_codes() {
        let exit_codes = r#"Exit Codes:
    0 = No errors
    1 = Encountered operational errors e.g. an IO error
    2 = Failed to reformat code because of parsing errors
    3 = Code is valid, but it is impossible to format it properly
    4 = Formatted code differs from existing code (write-mode diff only)"#;
        println!("{}", exit_codes);
    }
}

#[derive(Clone, Copy, Debug)]
enum Timer {
    Initialized(u64),
    DoneParsing(u64, u64),
    DoneFormatting(u64, u64, u64),
}

impl Default for Timer {
    fn default() -> Self {
        Timer::Initialized(precise_time_ns())
    }
}

impl Timer {
    fn done_parsing(self) -> Self {
        match self {
            Timer::Initialized(init_time) => Timer::DoneParsing(init_time, precise_time_ns()),
            _ => panic!("Timer can only transition to DoneParsing from Initialized state"),
        }
    }

    fn done_formatting(self) -> Self {
        match self {
            Timer::DoneParsing(init_time, parse_time) => {
                Timer::DoneFormatting(init_time, parse_time, precise_time_ns())
            }
            _ => panic!("Timer can only transition to DoneFormatting from DoneParsing state"),
        }
    }
}
