use std::{io, io::prelude::Write};

use super::OutputFormatter;
use crate::{
    bench::fmt_bench_samples,
    console::{ConsoleTestState, OutputLocation},
    term,
    test_result::TestResult,
    time,
    types::NamePadding,
    types::TestDesc,
};

// insert a '\n' after 100 tests in quiet mode
const QUIET_MODE_MAX_COLUMN: usize = 100;

pub(crate) struct TerseFormatter<T> {
    out: OutputLocation<T>,
    use_color: bool,
    is_multithreaded: bool,
    /// Number of columns to fill when aligning names
    max_name_len: usize,

    test_count: usize,
    total_test_count: usize,
}

impl<T: Write> TerseFormatter<T> {
    pub fn new(
        out: OutputLocation<T>,
        use_color: bool,
        max_name_len: usize,
        is_multithreaded: bool,
    ) -> Self {
        TerseFormatter {
            out,
            use_color,
            max_name_len,
            is_multithreaded,
            test_count: 0,
            total_test_count: 0, // initialized later, when write_run_start is called
        }
    }

    pub fn write_ok(&mut self) -> io::Result<()> {
        self.write_short_result(".", term::color::GREEN)
    }

    pub fn write_failed(&mut self) -> io::Result<()> {
        self.write_short_result("F", term::color::RED)
    }

    pub fn write_ignored(&mut self) -> io::Result<()> {
        self.write_short_result("i", term::color::YELLOW)
    }

    pub fn write_allowed_fail(&mut self) -> io::Result<()> {
        self.write_short_result("a", term::color::YELLOW)
    }

    pub fn write_bench(&mut self) -> io::Result<()> {
        self.write_pretty("bench", term::color::CYAN)
    }

    pub fn write_short_result(
        &mut self,
        result: &str,
        color: term::color::Color,
    ) -> io::Result<()> {
        self.write_pretty(result, color)?;
        if self.test_count % QUIET_MODE_MAX_COLUMN == QUIET_MODE_MAX_COLUMN - 1 {
            // we insert a new line every 100 dots in order to flush the
            // screen when dealing with line-buffered output (e.g., piping to
            // `stamp` in the rust CI).
            let out = format!(" {}/{}\n", self.test_count + 1, self.total_test_count);
            self.write_plain(&out)?;
        }

        self.test_count += 1;
        Ok(())
    }

    pub fn write_pretty(&mut self, word: &str, color: term::color::Color) -> io::Result<()> {
        match self.out {
            OutputLocation::Pretty(ref mut term) => {
                if self.use_color {
                    term.fg(color)?;
                }
                term.write_all(word.as_bytes())?;
                if self.use_color {
                    term.reset()?;
                }
                term.flush()
            }
            OutputLocation::Raw(ref mut stdout) => {
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
                stdouts.push_str(&format!("---- {} stdout ----\n", f.name));
                let output = String::from_utf8_lossy(stdout);
                stdouts.push_str(&output);
                stdouts.push('\n');
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
                fail_out.push_str(&format!("---- {} stdout ----\n", f.name));
                let output = String::from_utf8_lossy(stdout);
                fail_out.push_str(&output);
                fail_out.push('\n');
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

    fn write_test_name(&mut self, desc: &TestDesc) -> io::Result<()> {
        let name = desc.padded_name(self.max_name_len, desc.name.padding());
        if let Some(test_mode) = desc.test_mode() {
            self.write_plain(&format!("test {} - {} ... ", name, test_mode))?;
        } else {
            self.write_plain(&format!("test {} ... ", name))?;
        }

        Ok(())
    }
}

impl<T: Write> OutputFormatter for TerseFormatter<T> {
    fn write_run_start(&mut self, test_count: usize) -> io::Result<()> {
        self.total_test_count = test_count;
        let noun = if test_count != 1 { "tests" } else { "test" };
        self.write_plain(&format!("\nrunning {} {}\n", test_count, noun))
    }

    fn write_test_start(&mut self, desc: &TestDesc) -> io::Result<()> {
        // Remnants from old libtest code that used the padding value
        // in order to indicate benchmarks.
        // When running benchmarks, terse-mode should still print their name as if
        // it is the Pretty formatter.
        if !self.is_multithreaded && desc.name.padding() == NamePadding::PadOnRight {
            self.write_test_name(desc)?;
        }

        Ok(())
    }

    fn write_result(
        &mut self,
        desc: &TestDesc,
        result: &TestResult,
        _: Option<&time::TestExecTime>,
        _: &[u8],
        _: &ConsoleTestState,
    ) -> io::Result<()> {
        match *result {
            TestResult::TrOk => self.write_ok(),
            TestResult::TrFailed | TestResult::TrFailedMsg(_) | TestResult::TrTimedFail => {
                self.write_failed()
            }
            TestResult::TrIgnored => self.write_ignored(),
            TestResult::TrAllowedFail => self.write_allowed_fail(),
            TestResult::TrBench(ref bs) => {
                if self.is_multithreaded {
                    self.write_test_name(desc)?;
                }
                self.write_bench()?;
                self.write_plain(&format!(": {}\n", fmt_bench_samples(bs)))
            }
        }
    }

    fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()> {
        self.write_plain(&format!(
            "test {} has been running for over {} seconds\n",
            desc.name,
            time::TEST_WARN_TIMEOUT_S
        ))
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
                ". {} passed; {} failed ({} allowed); {} ignored; {} measured; {} filtered out",
                state.passed,
                state.failed + state.allowed_fail,
                state.allowed_fail,
                state.ignored,
                state.measured,
                state.filtered_out
            )
        } else {
            format!(
                ". {} passed; {} failed; {} ignored; {} measured; {} filtered out",
                state.passed, state.failed, state.ignored, state.measured, state.filtered_out
            )
        };

        self.write_plain(&s)?;

        if let Some(ref exec_time) = state.exec_time {
            let time_str = format!("; finished in {}", exec_time);
            self.write_plain(&time_str)?;
        }

        self.write_plain("\n\n")?;

        Ok(success)
    }
}
