use std::char;
use std::fmt::Write;

use crate::{Float, Generator};

/// Number of decimal digits to check (all of them).
const MAX_DIGIT: u32 = 9;
/// Test with this many decimals in the string.
const MAX_DECIMALS: usize = 410;
const PREFIX: &str = "0.";

/// Test e.g. `0.1`, `0.11`, `0.111`, `0.1111`, ..., `0.2`, `0.22`, ...
pub struct RepeatingDecimal {
    digit: u32,
    buf: String,
}

impl<F: Float> Generator<F> for RepeatingDecimal {
    const NAME: &'static str = "repeating decimal";
    const SHORT_NAME: &'static str = "dec rep";

    type WriteCtx = String;

    fn total_tests() -> u64 {
        u64::from(MAX_DIGIT + 1) * u64::try_from(MAX_DECIMALS + 1).unwrap() + 1
    }

    fn new() -> Self {
        Self { digit: 0, buf: PREFIX.to_owned() }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        *s = ctx;
    }
}

impl Iterator for RepeatingDecimal {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.digit > MAX_DIGIT {
            return None;
        }

        let digit = self.digit;
        let inc_digit = self.buf.len() - PREFIX.len() > MAX_DECIMALS;

        if inc_digit {
            // Reset the string
            self.buf.clear();
            self.digit += 1;
            self.buf.write_str(PREFIX).unwrap();
        }

        self.buf.push(char::from_digit(digit, 10).unwrap());
        Some(self.buf.clone())
    }
}
