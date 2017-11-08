// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;

pub(crate) trait OutputFormatter {
    fn write_run_start(&mut self, len: usize) -> io::Result<()>;
    fn write_test_start(&mut self,
        test: &TestDesc,
        align: NamePadding,
        max_name_len: usize) -> io::Result<()>;
    fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()>;
    fn write_result(&mut self, result: &TestResult) -> io::Result<()>;
    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool>;
}

pub(crate) struct HumanFormatter<T> {
    out: OutputLocation<T>,
    terse: bool,
    use_color: bool,
    test_count: usize,
}

impl<T: Write> HumanFormatter<T> {
    pub fn new(out: OutputLocation<T>, use_color: bool, terse: bool) -> Self {
        HumanFormatter {
            out,
            terse,
            use_color,
            test_count: 0,
        }
    }

    #[cfg(test)]
    pub fn output_location(&self) -> &OutputLocation<T> {
        &self.out
    }

    pub fn write_ok(&mut self) -> io::Result<()> {
        self.write_short_result("ok", ".", term::color::GREEN)
    }

    pub fn write_failed(&mut self) -> io::Result<()> {
        self.write_short_result("FAILED", "F", term::color::RED)
    }

    pub fn write_ignored(&mut self) -> io::Result<()> {
        self.write_short_result("ignored", "i", term::color::YELLOW)
    }

    pub fn write_allowed_fail(&mut self) -> io::Result<()> {
        self.write_short_result("FAILED (allowed)", "a", term::color::YELLOW)
    }

    pub fn write_bench(&mut self) -> io::Result<()> {
        self.write_pretty("bench", term::color::CYAN)
    }

    pub fn write_short_result(&mut self, verbose: &str, quiet: &str, color: term::color::Color)
                              -> io::Result<()> {
        if self.terse {
            self.write_pretty(quiet, color)?;
            if self.test_count % QUIET_MODE_MAX_COLUMN == QUIET_MODE_MAX_COLUMN - 1 {
                // we insert a new line every 100 dots in order to flush the
                // screen when dealing with line-buffered output (e.g. piping to
                // `stamp` in the rust CI).
                self.write_plain("\n")?;
            }
            
            self.test_count += 1;
            Ok(())
        } else {
            self.write_pretty(verbose, color)?;
            self.write_plain("\n")
        }
    }

    pub fn write_pretty(&mut self, word: &str, color: term::color::Color) -> io::Result<()> {
        match self.out {
            Pretty(ref mut term) => {
                if self.use_color {
                    term.fg(color)?;
                }
                term.write_all(word.as_bytes())?;
                if self.use_color {
                    term.reset()?;
                }
                term.flush()
            }
            Raw(ref mut stdout) => {
                stdout.write_all(word.as_bytes())?;
                stdout.flush()
            }
        }
    }

    pub fn write_plain<S: AsRef<str>>(&mut self, s: S) -> io::Result<()> {
        let s = s.as_ref();
        self.out.write_all(s.as_bytes())?;
        self.out.flush()
    }

    pub fn write_outputs(&mut self, state: &ConsoleTestState) -> io::Result<()> {
        self.write_plain("\nsuccesses:\n")?;
        let mut successes = Vec::new();
        let mut stdouts = String::new();
        for &(ref f, ref stdout) in &state.not_failures {
            successes.push(f.name.to_string());
            if !stdout.is_empty() {
                stdouts.push_str(&format!("---- {} stdout ----\n\t", f.name));
                let output = String::from_utf8_lossy(stdout);
                stdouts.push_str(&output);
                stdouts.push_str("\n");
            }
        }
        if !stdouts.is_empty() {
            self.write_plain("\n")?;
            self.write_plain(&stdouts)?;
        }

        self.write_plain("\nsuccesses:\n")?;
        successes.sort();
        for name in &successes {
            self.write_plain(&format!("    {}\n", name))?;
        }
        Ok(())
    }

    pub fn write_failures(&mut self, state: &ConsoleTestState) -> io::Result<()> {
        self.write_plain("\nfailures:\n")?;
        let mut failures = Vec::new();
        let mut fail_out = String::new();
        for &(ref f, ref stdout) in &state.failures {
            failures.push(f.name.to_string());
            if !stdout.is_empty() {
                fail_out.push_str(&format!("---- {} stdout ----\n\t", f.name));
                let output = String::from_utf8_lossy(stdout);
                fail_out.push_str(&output);
                fail_out.push_str("\n");
            }
        }
        if !fail_out.is_empty() {
            self.write_plain("\n")?;
            self.write_plain(&fail_out)?;
        }

        self.write_plain("\nfailures:\n")?;
        failures.sort();
        for name in &failures {
            self.write_plain(&format!("    {}\n", name))?;
        }
        Ok(())
    }
}

impl<T: Write> OutputFormatter for HumanFormatter<T> {
    fn write_run_start(&mut self, len: usize) -> io::Result<()> {
        let noun = if len != 1 {
            "tests"
        } else {
            "test"
        };
        self.write_plain(&format!("\nrunning {} {}\n", len, noun))
    }

    fn write_test_start(&mut self,
                        test: &TestDesc,
                        align: NamePadding,
                        max_name_len: usize) -> io::Result<()> {
        if self.terse && align != PadOnRight {
            Ok(())
        }
        else {
            let name = test.padded_name(max_name_len, align);
            self.write_plain(&format!("test {} ... ", name))
        }
    }

    fn write_result(&mut self, result: &TestResult) -> io::Result<()> {
        match *result {
            TrOk => self.write_ok(),
            TrFailed | TrFailedMsg(_) => self.write_failed(),
            TrIgnored => self.write_ignored(),
            TrAllowedFail => self.write_allowed_fail(),
            TrBench(ref bs) => {
                self.write_bench()?;
                self.write_plain(&format!(": {}\n", fmt_bench_samples(bs)))
            }
        }
    }

    fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()> {
        self.write_plain(&format!("test {} has been running for over {} seconds\n",
                                  desc.name,
                                  TEST_WARN_TIMEOUT_S))
    }

    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool> {
        if state.options.display_output {
            self.write_outputs(state)?;
        }
        let success = state.failed == 0;
        if !success {
            self.write_failures(state)?;
        }

        self.write_plain("\ntest result: ")?;

        if success {
            // There's no parallelism at this point so it's safe to use color
            self.write_pretty("ok", term::color::GREEN)?;
        } else {
            self.write_pretty("FAILED", term::color::RED)?;
        }

        let s = if state.allowed_fail > 0 {
            format!(
                ". {} passed; {} failed ({} allowed); {} ignored; {} measured; {} filtered out\n\n",
                state.passed,
                state.failed + state.allowed_fail,
                state.allowed_fail,
                state.ignored,
                state.measured,
                state.filtered_out)
        } else {
            format!(
                ". {} passed; {} failed; {} ignored; {} measured; {} filtered out\n\n",
                state.passed,
                state.failed,
                state.ignored,
                state.measured,
                state.filtered_out)
        };

        self.write_plain(&s)?;

        Ok(success)
    }
}
