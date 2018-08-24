use super::*;

pub(crate) struct PrettyFormatter<T> {
    out: OutputLocation<T>,
    use_color: bool,

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
    ) -> Self {
        PrettyFormatter {
            out,
            use_color,
            max_name_len,
            is_multithreaded,
        }
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

    pub fn write_bench(&mut self) -> io::Result<()> {
        self.write_pretty("bench", term::color::CYAN)
    }

    pub fn write_short_result(
        &mut self,
        result: &str,
        color: term::color::Color,
    ) -> io::Result<()> {
        self.write_pretty(result, color)?;
        self.write_plain("\n")
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

    pub fn write_successes(&mut self, state: &ConsoleTestState) -> io::Result<()> {
        self.write_plain("\nsuccesses:\n")?;
        let mut successes = Vec::new();
        let mut stdouts = String::new();
        for &(ref f, ref stdout) in &state.not_failures {
            successes.push(f.name.to_string());
            if !stdout.is_empty() {
                stdouts.push_str(&format!("---- {} stdout ----\n", f.name));
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
                fail_out.push_str(&format!("---- {} stdout ----\n", f.name));
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

    fn write_test_name(&mut self, desc: &TestDesc) -> io::Result<()> {
        let name = desc.padded_name(self.max_name_len, desc.name.padding());
        self.write_plain(&format!("test {} ... ", name))?;

        Ok(())
    }
}

impl<T: Write> OutputFormatter for PrettyFormatter<T> {
    fn write_run_start(&mut self, test_count: usize) -> io::Result<()> {
        let noun = if test_count != 1 { "tests" } else { "test" };
        self.write_plain(&format!("\nrunning {} {}\n", test_count, noun))
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

    fn write_result(&mut self, desc: &TestDesc, result: &TestResult, _: &[u8]) -> io::Result<()> {
        if self.is_multithreaded {
            self.write_test_name(desc)?;
        }

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
        if self.is_multithreaded {
            self.write_test_name(desc)?;
        }

        self.write_plain(&format!(
            "test {} has been running for over {} seconds\n",
            desc.name, TEST_WARN_TIMEOUT_S
        ))
    }

    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool> {
        if state.options.display_output {
            self.write_successes(state)?;
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
                state.filtered_out
            )
        } else {
            format!(
                ". {} passed; {} failed; {} ignored; {} measured; {} filtered out\n\n",
                state.passed, state.failed, state.ignored, state.measured, state.filtered_out
            )
        };

        self.write_plain(&s)?;

        Ok(success)
    }
}
