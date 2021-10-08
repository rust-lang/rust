use std::{io, io::prelude::Write};

use super::OutputFormatter;
use crate::{
    bench::fmt_bench_samples,
    console::{ConsoleTestState, OutputLocation},
    term,
    test_result::TestResult,
    time,
    types::TestDesc,
};

pub(crate) struct PrettyFormatter<T> {
    out: OutputLocation<T>,
    use_color: bool,
    time_options: Option<time::TestTimeOptions>,

    /// Number of columns to fill when aligning names
    max_name_len: usize,

    is_multithreaded: bool,
}

impl<T: Write> PrettyFormatter<T> {
    pub fn new(
        out: OutputLocation<T>,
        use_color: bool,
        max_name_len: usize,
        is_multithreaded: bool,
        time_options: Option<time::TestTimeOptions>,
    ) -> Self {
        PrettyFormatter { out, use_color, max_name_len, is_multithreaded, time_options }
    }

    #[cfg(test)]
    pub fn output_location(&self) -> &OutputLocation<T> {
        &self.out
    }

    pub fn write_ok(&mut self) -> io::Result<()> {
        self.write_short_result("ok", term::color::GREEN)
    }

    pub fn write_failed(&mut self) -> io::Result<()> {
        self.write_short_result("FAILED", term::color::RED)
    }

    pub fn write_ignored(&mut self) -> io::Result<()> {
        self.write_short_result("ignored", term::color::YELLOW)
    }

    pub fn write_allowed_fail(&mut self) -> io::Result<()> {
        self.write_short_result("FAILED (allowed)", term::color::YELLOW)
    }

    pub fn write_time_failed(&mut self) -> io::Result<()> {
        self.write_short_result("FAILED (time limit exceeded)", term::color::RED)
    }

    pub fn write_bench(&mut self) -> io::Result<()> {
        self.write_pretty("bench", term::color::CYAN)
    }

    pub fn write_short_result(
        &mut self,
        result: &str,
        color: term::color::Color,
    ) -> io::Result<()> {
        self.write_pretty(result, color)
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

    fn write_time(
        &mut self,
        desc: &TestDesc,
        exec_time: Option<&time::TestExecTime>,
    ) -> io::Result<()> {
        if let (Some(opts), Some(time)) = (self.time_options, exec_time) {
            let time_str = format!(" <{}>", time);

            let color = if opts.colored {
                if opts.is_critical(desc, time) {
                    Some(term::color::RED)
                } else if opts.is_warn(desc, time) {
                    Some(term::color::YELLOW)
                } else {
                    None
                }
            } else {
                None
            };

            match color {
                Some(color) => self.write_pretty(&time_str, color)?,
                None => self.write_plain(&time_str)?,
            }
        }

        Ok(())
    }

    fn write_results(
        &mut self,
        inputs: &Vec<(TestDesc, Vec<u8>)>,
        results_type: &str,
    ) -> io::Result<()> {
        let results_out_str = format!("\n{}:\n", results_type);

        self.write_plain(&results_out_str)?;

        let mut results = Vec::new();
        let mut stdouts = String::new();
        for &(ref f, ref stdout) in inputs {
            results.push(f.name.to_string());
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

        self.write_plain(&results_out_str)?;
        results.sort();
        for name in &results {
            self.write_plain(&format!("    {}\n", name))?;
        }
        Ok(())
    }

    pub fn write_successes(&mut self, state: &ConsoleTestState) -> io::Result<()> {
        self.write_results(&state.not_failures, "successes")
    }

    pub fn write_failures(&mut self, state: &ConsoleTestState) -> io::Result<()> {
        self.write_results(&state.failures, "failures")
    }

    pub fn write_time_failures(&mut self, state: &ConsoleTestState) -> io::Result<()> {
        self.write_results(&state.time_failures, "failures (time limit exceeded)")
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

impl<T: Write> OutputFormatter for PrettyFormatter<T> {
    fn write_run_start(&mut self, test_count: usize, shuffle_seed: Option<u64>) -> io::Result<()> {
        let noun = if test_count != 1 { "tests" } else { "test" };
        let shuffle_seed_msg = if let Some(shuffle_seed) = shuffle_seed {
            format!(" (shuffle seed: {})", shuffle_seed)
        } else {
            String::new()
        };
        self.write_plain(&format!("\nrunning {} {}{}\n", test_count, noun, shuffle_seed_msg))
    }

    fn write_test_start(&mut self, desc: &TestDesc) -> io::Result<()> {
        // When running tests concurrently, we should not print
        // the test's name as the result will be mis-aligned.
        // When running the tests serially, we print the name here so
        // that the user can see which test hangs.
        if !self.is_multithreaded {
            self.write_test_name(desc)?;
        }

        Ok(())
    }

    fn write_result(
        &mut self,
        desc: &TestDesc,
        result: &TestResult,
        exec_time: Option<&time::TestExecTime>,
        _: &[u8],
        _: &ConsoleTestState,
    ) -> io::Result<()> {
        if self.is_multithreaded {
            self.write_test_name(desc)?;
        }

        match *result {
            TestResult::TrOk => self.write_ok()?,
            TestResult::TrFailed | TestResult::TrFailedMsg(_) => self.write_failed()?,
            TestResult::TrIgnored => self.write_ignored()?,
            TestResult::TrAllowedFail => self.write_allowed_fail()?,
            TestResult::TrBench(ref bs) => {
                self.write_bench()?;
                self.write_plain(&format!(": {}", fmt_bench_samples(bs)))?;
            }
            TestResult::TrTimedFail => self.write_time_failed()?,
        }

        self.write_time(desc, exec_time)?;
        self.write_plain("\n")
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
            self.write_successes(state)?;
        }
        let success = state.failed == 0;
        if !success {
            if !state.failures.is_empty() {
                self.write_failures(state)?;
            }

            if !state.time_failures.is_empty() {
                self.write_time_failures(state)?;
            }
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
