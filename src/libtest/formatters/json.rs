use super::*;

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
                ty, name, evt, extras
            ))
        } else {
            self.write_message(&*format!(
                r#"{{ "type": "{}", "name": "{}", "event": "{}" }}"#,
                ty, name, evt
            ))
        }
    }
}

impl<T: Write> OutputFormatter for JsonFormatter<T> {
    fn write_run_start(&mut self, test_count: usize) -> io::Result<()> {
        self.write_message(&*format!(
            r#"{{ "type": "suite", "event": "started", "test_count": {} }}"#,
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

            TrFailedMsg(ref m) => self.write_event(
                "test",
                desc.name.as_slice(),
                "failed",
                Some(format!(r#""message": "{}""#, EscapedString(m))),
            ),

            TrIgnored => self.write_event("test", desc.name.as_slice(), "ignored", None),

            TrAllowedFail => {
                self.write_event("test", desc.name.as_slice(), "allowed_failure", None)
            }

            TrBench(ref bs) => {
                let median = bs.ns_iter_summ.median as usize;
                let deviation = (bs.ns_iter_summ.max - bs.ns_iter_summ.min) as usize;

                let mbps = if bs.mb_s == 0 {
                    String::new()
                } else {
                    format!(r#", "mib_per_second": {}"#, bs.mb_s)
                };

                let line = format!(
                    "{{ \"type\": \"bench\", \
                     \"name\": \"{}\", \
                     \"median\": {}, \
                     \"deviation\": {}{} }}",
                    desc.name, median, deviation, mbps
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
             \"filtered_out\": {} }}",
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
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
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
