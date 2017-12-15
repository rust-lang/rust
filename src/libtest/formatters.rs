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
    fn write_run_start(&mut self, test_count: usize) -> io::Result<()>;
    fn write_test_start(&mut self, desc: &TestDesc) -> io::Result<()>;
    fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()>;
    fn write_result(
        &mut self,
        desc: &TestDesc,
        result: &TestResult,
        stdout: &[u8],
    ) -> io::Result<()>;
    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool>;
}

pub(crate) struct HumanFormatter<T> {
    out: OutputLocation<T>,
    terse: bool,
    use_color: bool,
    test_count: usize,

    /// Number of columns to fill when aligning names
    max_name_len: usize,

    is_multithreaded: bool,
}

impl<T: Write> HumanFormatter<T> {
    pub fn new(
        out: OutputLocation<T>,
        use_color: bool,
        terse: bool,
        max_name_len: usize,
        is_multithreaded: bool,
    ) -> Self {
        HumanFormatter {
            out,
            terse,
            use_color,
            test_count: 0,
            max_name_len,
            is_multithreaded,
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

    pub fn write_short_result(
        &mut self,
        verbose: &str,
        quiet: &str,
        color: term::color::Color,
    ) -> io::Result<()> {
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

    fn write_test_name(&mut self, desc: &TestDesc) -> io::Result<()> {
        if !(self.terse && desc.name.padding() != PadOnRight) {
            let name = desc.padded_name(self.max_name_len, desc.name.padding());
            self.write_plain(&format!("test {} ... ", name))?;
        }

        Ok(())
    }
}

impl<T: Write> OutputFormatter for HumanFormatter<T> {
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
            desc.name,
            TEST_WARN_TIMEOUT_S
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
                state.passed,
                state.failed,
                state.ignored,
                state.measured,
                state.filtered_out
            )
        };

        self.write_plain(&s)?;

        Ok(success)
    }
}

pub(crate) struct JsonFormatter<T> {
    out: OutputLocation<T>,
}

impl<T: Write> JsonFormatter<T> {
    pub fn new(out: OutputLocation<T>) -> Self {
        Self { out }
    }

    fn write_message(&mut self, s: &str) -> io::Result<()> {
        assert!(!s.contains('\n'));

        self.out.write_all(s.as_ref())?;
        self.out.write_all(b"\n")
    }

    fn write_event(
        &mut self,
        ty: &str,
        name: &str,
        evt: &str,
        extra: Option<String>,
    ) -> io::Result<()> {
        if let Some(extras) = extra {
            self.write_message(&*format!(
                r#"{{ "type": "{}", "name": "{}", "event": "{}", {} }}"#,
                ty,
                name,
                evt,
                extras
            ))
        } else {
            self.write_message(&*format!(
                r#"{{ "type": "{}", "name": "{}", "event": "{}" }}"#,
                ty,
                name,
                evt
            ))
        }
    }
}

impl<T: Write> OutputFormatter for JsonFormatter<T> {
    fn write_run_start(&mut self, test_count: usize) -> io::Result<()> {
        self.write_message(&*format!(
            r#"{{ "type": "suite", "event": "started", "test_count": "{}" }}"#,
            test_count
        ))
    }

    fn write_test_start(&mut self, desc: &TestDesc) -> io::Result<()> {
        self.write_message(&*format!(
            r#"{{ "type": "test", "event": "started", "name": "{}" }}"#,
            desc.name
        ))
    }

    fn write_result(
        &mut self,
        desc: &TestDesc,
        result: &TestResult,
        stdout: &[u8],
    ) -> io::Result<()> {
        match *result {
            TrOk => self.write_event("test", desc.name.as_slice(), "ok", None),

            TrFailed => {
                let extra_data = if stdout.len() > 0 {
                    Some(format!(
                        r#""stdout": "{}""#,
                        EscapedString(String::from_utf8_lossy(stdout))
                    ))
                } else {
                    None
                };

                self.write_event("test", desc.name.as_slice(), "failed", extra_data)
            }

            TrFailedMsg(ref m) => {
                self.write_event(
                    "test",
                    desc.name.as_slice(),
                    "failed",
                    Some(format!(r#""message": "{}""#, EscapedString(m))),
                )
            }

            TrIgnored => self.write_event("test", desc.name.as_slice(), "ignored", None),

            TrAllowedFail => {
                self.write_event("test", desc.name.as_slice(), "allowed_failure", None)
            }

            TrBench(ref bs) => {
                let median = bs.ns_iter_summ.median as usize;
                let deviation = (bs.ns_iter_summ.max - bs.ns_iter_summ.min) as usize;

                let mbps = if bs.mb_s == 0 {
                    "".into()
                } else {
                    format!(r#", "mib_per_second": {}"#, bs.mb_s)
                };

                let line = format!(
                    "{{ \"type\": \"bench\", \
                                \"name\": \"{}\", \
                                \"median\": {}, \
                                \"deviation\": {}{} }}",
                    desc.name,
                    median,
                    deviation,
                    mbps
                );

                self.write_message(&*line)
            }
        }
    }

    fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()> {
        self.write_message(&*format!(
            r#"{{ "type": "test", "event": "timeout", "name": "{}" }}"#,
            desc.name
        ))
    }

    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool> {

        self.write_message(&*format!(
            "{{ \"type\": \"suite\", \
            \"event\": \"{}\", \
            \"passed\": {}, \
            \"failed\": {}, \
            \"allowed_fail\": {}, \
            \"ignored\": {}, \
            \"measured\": {}, \
            \"filtered_out\": \"{}\" }}",
            if state.failed == 0 { "ok" } else { "failed" },
            state.passed,
            state.failed + state.allowed_fail,
            state.allowed_fail,
            state.ignored,
            state.measured,
            state.filtered_out
        ))?;

        Ok(state.failed == 0)
    }
}

/// A formatting utility used to print strings with characters in need of escaping.
/// Base code taken form `libserialize::json::escape_str`
struct EscapedString<S: AsRef<str>>(S);

impl<S: AsRef<str>> ::std::fmt::Display for EscapedString<S> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        let mut start = 0;

        for (i, byte) in self.0.as_ref().bytes().enumerate() {
            let escaped = match byte {
                b'"' => "\\\"",
                b'\\' => "\\\\",
                b'\x00' => "\\u0000",
                b'\x01' => "\\u0001",
                b'\x02' => "\\u0002",
                b'\x03' => "\\u0003",
                b'\x04' => "\\u0004",
                b'\x05' => "\\u0005",
                b'\x06' => "\\u0006",
                b'\x07' => "\\u0007",
                b'\x08' => "\\b",
                b'\t' => "\\t",
                b'\n' => "\\n",
                b'\x0b' => "\\u000b",
                b'\x0c' => "\\f",
                b'\r' => "\\r",
                b'\x0e' => "\\u000e",
                b'\x0f' => "\\u000f",
                b'\x10' => "\\u0010",
                b'\x11' => "\\u0011",
                b'\x12' => "\\u0012",
                b'\x13' => "\\u0013",
                b'\x14' => "\\u0014",
                b'\x15' => "\\u0015",
                b'\x16' => "\\u0016",
                b'\x17' => "\\u0017",
                b'\x18' => "\\u0018",
                b'\x19' => "\\u0019",
                b'\x1a' => "\\u001a",
                b'\x1b' => "\\u001b",
                b'\x1c' => "\\u001c",
                b'\x1d' => "\\u001d",
                b'\x1e' => "\\u001e",
                b'\x1f' => "\\u001f",
                b'\x7f' => "\\u007f",
                _ => {
                    continue;
                }
            };

            if start < i {
                f.write_str(&self.0.as_ref()[start..i])?;
            }

            f.write_str(escaped)?;

            start = i + 1;
        }

        if start != self.0.as_ref().len() {
            f.write_str(&self.0.as_ref()[start..])?;
        }

        Ok(())
    }
}
