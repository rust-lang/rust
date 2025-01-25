use std::io::prelude::Write;
use std::io::{self};
use std::time::Duration;

use super::OutputFormatter;
use crate::console::{ConsoleTestDiscoveryState, ConsoleTestState, OutputLocation};
use crate::test_result::TestResult;
use crate::time;
use crate::types::{TestDesc, TestType};

pub(crate) struct JunitFormatter<T> {
    out: OutputLocation<T>,
    results: Vec<(TestDesc, TestResult, Duration, Vec<u8>)>,
}

impl<T: Write> JunitFormatter<T> {
    pub(crate) fn new(out: OutputLocation<T>) -> Self {
        Self { out, results: Vec::new() }
    }

    fn write_message(&mut self, s: &str) -> io::Result<()> {
        assert!(!s.contains('\n'));

        self.out.write_all(s.as_ref())
    }
}

fn str_to_cdata(s: &str) -> String {
    // Drop the stdout in a cdata. Unfortunately, you can't put either of `]]>` or
    // `<?'` in a CDATA block, so the escaping gets a little weird.
    let escaped_output = s.replace("]]>", "]]]]><![CDATA[>");
    let escaped_output = escaped_output.replace("<?", "<]]><![CDATA[?");
    // We also smuggle newlines as &#xa so as to keep all the output on one line
    let escaped_output = escaped_output.replace('\n', "]]>&#xA;<![CDATA[");
    // Prune empty CDATA blocks resulting from any escaping
    let escaped_output = escaped_output.replace("<![CDATA[]]>", "");
    format!("<![CDATA[{}]]>", escaped_output)
}

impl<T: Write> OutputFormatter for JunitFormatter<T> {
    fn write_discovery_start(&mut self) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::NotFound, "Not yet implemented!"))
    }

    fn write_test_discovered(&mut self, _desc: &TestDesc, _test_type: &str) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::NotFound, "Not yet implemented!"))
    }

    fn write_discovery_finish(&mut self, _state: &ConsoleTestDiscoveryState) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::NotFound, "Not yet implemented!"))
    }

    fn write_run_start(
        &mut self,
        _test_count: usize,
        _shuffle_seed: Option<u64>,
    ) -> io::Result<()> {
        // We write xml header on run start
        self.write_message("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    }

    fn write_test_start(&mut self, _desc: &TestDesc) -> io::Result<()> {
        // We do not output anything on test start.
        Ok(())
    }

    fn write_timeout(&mut self, _desc: &TestDesc) -> io::Result<()> {
        // We do not output anything on test timeout.
        Ok(())
    }

    fn write_result(
        &mut self,
        desc: &TestDesc,
        result: &TestResult,
        exec_time: Option<&time::TestExecTime>,
        stdout: &[u8],
        _state: &ConsoleTestState,
    ) -> io::Result<()> {
        // Because the testsuite node holds some of the information as attributes, we can't write it
        // until all of the tests have finished. Instead of writing every result as they come in, we add
        // them to a Vec and write them all at once when run is complete.
        let duration = exec_time.map(|t| t.0).unwrap_or_default();
        self.results.push((desc.clone(), result.clone(), duration, stdout.to_vec()));
        Ok(())
    }
    fn write_run_finish(&mut self, state: &ConsoleTestState) -> io::Result<bool> {
        self.write_message("<testsuites>")?;

        self.write_message(&format!(
            "<testsuite name=\"test\" package=\"test\" id=\"0\" \
             errors=\"0\" \
             failures=\"{}\" \
             tests=\"{}\" \
             skipped=\"{}\" \
             >",
            state.failed, state.total, state.ignored
        ))?;
        for (desc, result, duration, stdout) in std::mem::take(&mut self.results) {
            let (class_name, test_name) = parse_class_name(&desc);
            match result {
                TestResult::TrIgnored => { /* no-op */ }
                TestResult::TrFailed => {
                    self.write_message(&format!(
                        "<testcase classname=\"{}\" \
                         name=\"{}\" time=\"{}\">",
                        class_name,
                        test_name,
                        duration.as_secs_f64()
                    ))?;
                    self.write_message("<failure type=\"assert\"/>")?;
                    if !stdout.is_empty() {
                        self.write_message("<system-out>")?;
                        self.write_message(&str_to_cdata(&String::from_utf8_lossy(&stdout)))?;
                        self.write_message("</system-out>")?;
                    }
                    self.write_message("</testcase>")?;
                }

                TestResult::TrFailedMsg(ref m) => {
                    self.write_message(&format!(
                        "<testcase classname=\"{}\" \
                         name=\"{}\" time=\"{}\">",
                        class_name,
                        test_name,
                        duration.as_secs_f64()
                    ))?;
                    self.write_message(&format!("<failure message=\"{m}\" type=\"assert\"/>"))?;
                    if !stdout.is_empty() {
                        self.write_message("<system-out>")?;
                        self.write_message(&str_to_cdata(&String::from_utf8_lossy(&stdout)))?;
                        self.write_message("</system-out>")?;
                    }
                    self.write_message("</testcase>")?;
                }

                TestResult::TrTimedFail => {
                    self.write_message(&format!(
                        "<testcase classname=\"{}\" \
                         name=\"{}\" time=\"{}\">",
                        class_name,
                        test_name,
                        duration.as_secs_f64()
                    ))?;
                    self.write_message("<failure type=\"timeout\"/>")?;
                    self.write_message("</testcase>")?;
                }

                TestResult::TrBench(ref b) => {
                    self.write_message(&format!(
                        "<testcase classname=\"benchmark::{}\" \
                         name=\"{}\" time=\"{}\" />",
                        class_name, test_name, b.ns_iter_summ.sum
                    ))?;
                }

                TestResult::TrOk => {
                    self.write_message(&format!(
                        "<testcase classname=\"{}\" \
                         name=\"{}\" time=\"{}\"",
                        class_name,
                        test_name,
                        duration.as_secs_f64()
                    ))?;
                    if stdout.is_empty() || !state.options.display_output {
                        self.write_message("/>")?;
                    } else {
                        self.write_message("><system-out>")?;
                        self.write_message(&str_to_cdata(&String::from_utf8_lossy(&stdout)))?;
                        self.write_message("</system-out>")?;
                        self.write_message("</testcase>")?;
                    }
                }
            }
        }
        self.write_message("<system-out/>")?;
        self.write_message("<system-err/>")?;
        self.write_message("</testsuite>")?;
        self.write_message("</testsuites>")?;

        self.out.write_all(b"\n")?;

        Ok(state.failed == 0)
    }
}

fn parse_class_name(desc: &TestDesc) -> (String, String) {
    match desc.test_type {
        TestType::UnitTest => parse_class_name_unit(desc),
        TestType::DocTest => parse_class_name_doc(desc),
        TestType::IntegrationTest => parse_class_name_integration(desc),
        TestType::Unknown => (String::from("unknown"), String::from(desc.name.as_slice())),
    }
}

fn parse_class_name_unit(desc: &TestDesc) -> (String, String) {
    // Module path => classname
    // Function name => name
    let module_segments: Vec<&str> = desc.name.as_slice().split("::").collect();
    let (class_name, test_name) = match module_segments[..] {
        [test] => (String::from("crate"), String::from(test)),
        [ref path @ .., test] => (path.join("::"), String::from(test)),
        [..] => unreachable!(),
    };
    (class_name, test_name)
}

fn parse_class_name_doc(desc: &TestDesc) -> (String, String) {
    // File path => classname
    // Line # => test name
    let segments: Vec<&str> = desc.name.as_slice().split(" - ").collect();
    let (class_name, test_name) = match segments[..] {
        [file, line] => (String::from(file.trim()), String::from(line.trim())),
        [..] => unreachable!(),
    };
    (class_name, test_name)
}

fn parse_class_name_integration(desc: &TestDesc) -> (String, String) {
    (String::from("integration"), String::from(desc.name.as_slice()))
}
