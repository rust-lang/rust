use std::borrow::Cow;
use std::io;
use std::io::prelude::Write;

use super::OutputFormatter;
use crate::console::{ConsoleTestDiscoveryState, ConsoleTestState, OutputLocation};
use crate::test_result::TestResult;
use crate::time;
use crate::types::TestDesc;

pub(crate) struct JsonFormatter<T> {
    out: OutputLocation<T>,
}

impl<T: Write> JsonFormatter<T> {
    pub(crate) fn new(out: OutputLocation<T>) -> Self {
        Self { out }
    }

    fn writeln_message(&mut self, s: &str) -> io::Result<()> {
        // self.out will take a lock, but that lock is released when write_all returns. This
        // results in a race condition and json output may not end with a new line. We avoid this
        // by issuing `write_all` calls line-by-line.
        assert_eq!(s.chars().last(), Some('\n'));

        self.out.write_all(s.as_ref())
    }

    fn write_event(
        &mut self,
        ty: &str,
        name: &str,
        event: &str,
        exec_time: Option<&time::TestExecTime>,
        stdout: Option<Cow<'_, str>>,
        extra: Option<&str>,
    ) -> io::Result<()> {
        // A doc test's name includes a filename which must be escaped for correct json.
        let name = EscapedString(name);
        let exec_time_json = if let Some(exec_time) = exec_time {
            format!(r#", "exec_time": {}"#, exec_time.0.as_secs_f64())
        } else {
            String::from("")
        };
        let stdout_json = if let Some(stdout) = stdout {
            format!(r#", "stdout": "{}""#, EscapedString(stdout))
        } else {
            String::from("")
        };
        let extra_json =
            if let Some(extra) = extra { format!(r#", {extra}"#) } else { String::from("") };
        let newline = "\n";

        self.writeln_message(&format!(
                r#"{{ "type": "{ty}", "name": "{name}", "event": "{event}"{exec_time_json}{stdout_json}{extra_json} }}{newline}"#))
    }
}

impl<T: Write> OutputFormatter for JsonFormatter<T> {
    fn write_discovery_start(&mut self) -> io::Result<()> {
        self.writeln_message(concat!(r#"{ "type": "suite", "event": "discovery" }"#, "\n"))
    }

    fn write_test_discovered(&mut self, desc: &TestDesc, test_type: &str) -> io::Result<()> {
        let TestDesc {
            name,
            ignore,
            ignore_message,
            source_file,
            start_line,
            start_col,
            end_line,
            end_col,
            ..
        } = desc;

        let name = EscapedString(name.as_slice());
        let ignore_message = ignore_message.unwrap_or("");
        let source_path = EscapedString(source_file);
        let newline = "\n";

        self.writeln_message(&format!(
            r#"{{ "type": "{test_type}", "event": "discovered", "name": "{name}", "ignore": {ignore}, "ignore_message": "{ignore_message}", "source_path": "{source_path}", "start_line": {start_line}, "start_col": {start_col}, "end_line": {end_line}, "end_col": {end_col} }}{newline}"#
        ))
    }

    fn write_discovery_finish(&mut self, state: &ConsoleTestDiscoveryState) -> io::Result<()> {
        let ConsoleTestDiscoveryState { tests, benchmarks, ignored, .. } = state;

        let total = tests + benchmarks;
        let newline = "\n";
        self.writeln_message(&format!(
            r#"{{ "type": "suite", "event": "completed", "tests": {tests}, "benchmarks": {benchmarks}, "total": {total}, "ignored": {ignored} }}{newline}"#
            ))
    }

    fn write_run_start(&mut self, test_count: usize, shuffle_seed: Option<u64>) -> io::Result<()> {
        let shuffle_seed_json = if let Some(shuffle_seed) = shuffle_seed {
            format!(r#", "shuffle_seed": {shuffle_seed}"#)
        } else {
            String::new()
        };
        let newline = "\n";
        self.writeln_message(&format!(
            r#"{{ "type": "suite", "event": "started", "test_count": {test_count}{shuffle_seed_json} }}{newline}"#
            ))
    }

    fn write_test_start(&mut self, desc: &TestDesc) -> io::Result<()> {
        let name = EscapedString(desc.name.as_slice());
        let newline = "\n";
        self.writeln_message(&format!(
            r#"{{ "type": "test", "event": "started", "name": "{name}" }}{newline}"#
        ))
    }

    fn write_result(
        &mut self,
        desc: &TestDesc,
        result: &TestResult,
        exec_time: Option<&time::TestExecTime>,
        stdout: &[u8],
        state: &ConsoleTestState,
    ) -> io::Result<()> {
        let display_stdout = state.options.display_output || *result != TestResult::TrOk;
        let stdout = if display_stdout && !stdout.is_empty() {
            Some(String::from_utf8_lossy(stdout))
        } else {
            None
        };
        match *result {
            TestResult::TrOk => {
                self.write_event("test", desc.name.as_slice(), "ok", exec_time, stdout, None)
            }

            TestResult::TrFailed => {
                self.write_event("test", desc.name.as_slice(), "failed", exec_time, stdout, None)
            }

            TestResult::TrTimedFail => self.write_event(
                "test",
                desc.name.as_slice(),
                "failed",
                exec_time,
                stdout,
                Some(r#""reason": "time limit exceeded""#),
            ),

            TestResult::TrFailedMsg(ref m) => self.write_event(
                "test",
                desc.name.as_slice(),
                "failed",
                exec_time,
                stdout,
                Some(&*format!(r#""message": "{}""#, EscapedString(m))),
            ),

            TestResult::TrIgnored => self.write_event(
                "test",
                desc.name.as_slice(),
                "ignored",
                exec_time,
                stdout,
                desc.ignore_message
                    .map(|msg| format!(r#""message": "{}""#, EscapedString(msg)))
                    .as_deref(),
            ),

            TestResult::TrBench(ref bs) => {
                let median = bs.ns_iter_summ.median;
                let deviation = bs.ns_iter_summ.max - bs.ns_iter_summ.min;

                let mbps = if bs.mb_s == 0 {
                    String::new()
                } else {
                    format!(r#", "mib_per_second": {}"#, bs.mb_s)
                };
                let name = EscapedString(desc.name.as_slice());

                self.writeln_message(&format!(
                    "{{ \"type\": \"bench\", \
                     \"name\": \"{name}\", \
                     \"median\": {median}, \
                     \"deviation\": {deviation}{mbps} }}\n",
                ))
            }
        }
    }

    fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()> {
        let name = EscapedString(desc.name.as_slice());
        let newline = "\n";
        self.writeln_message(&format!(
            r#"{{ "type": "test", "event": "timeout", "name": "{name}" }}{newline}"#,
        ))
    }

    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool> {
        let event = if state.failed == 0 { "ok" } else { "failed" };
        let passed = state.passed;
        let failed = state.failed;
        let ignored = state.ignored;
        let measured = state.measured;
        let filtered_out = state.filtered_out;
        let exec_time_json = if let Some(ref exec_time) = state.exec_time {
            format!(r#", "exec_time": {}"#, exec_time.0.as_secs_f64())
        } else {
            String::from("")
        };
        let newline = "\n";

        self.writeln_message(&format!(
            r#"{{ "type": "suite", "event": "{event}", "passed": {passed}, "failed": {failed}, "ignored": {ignored}, "measured": {measured}, "filtered_out": {filtered_out}{exec_time_json} }}{newline}"#
        ))?;

        Ok(state.failed == 0)
    }

    fn write_merged_doctests_times(
        &mut self,
        total_time: f64,
        compilation_time: f64,
    ) -> io::Result<()> {
        let newline = "\n";
        self.writeln_message(&format!(
            r#"{{ "type": "report", "total_time": {total_time}, "compilation_time": {compilation_time} }}{newline}"#,
        ))
    }
}

/// A formatting utility used to print strings with characters in need of escaping.
/// Base code taken form `libserialize::json::escape_str`
struct EscapedString<S: AsRef<str>>(S);

impl<S: AsRef<str>> std::fmt::Display for EscapedString<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> ::std::fmt::Result {
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
