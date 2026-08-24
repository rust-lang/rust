//! Regression test to ensure that the output format is respected for doctests.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/144798>.

//@ ignore-cross-compile

use run_make_support::{rustdoc, serde_json};

fn run_test(edition: &str, format: Option<&str>) -> String {
    let mut r = rustdoc();
    r.input("file.rs").edition(edition).arg("--test");
    if let Some(format) = format {
        r.args(&[
            "--test-args",
            "-Zunstable-options",
            "--test-args",
            "--format",
            "--test-args",
            format,
        ]);
    }
    r.run().stdout_utf8()
}

fn check_json_output(edition: &str, expected_reports: usize) {
    let out = run_test(edition, Some("json"));
    let mut found_report = 0;
    for (line_nb, line) in out.lines().enumerate() {
        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(value) => {
                if value.get("type") == Some(&serde_json::json!("report")) {
                    found_report += 1;
                }
            }
            Err(error) => panic!(
                "failed for {edition} edition (json format) at line {}: non-JSON value: {error}\n\
                ====== output ======\n{out}",
                line_nb + 1,
            ),
        }
    }
    if found_report != expected_reports {
        panic!(
            "failed for {edition} edition (json format): expected {expected_reports} doctest \
             time `report`, found {found_report}\n====== output ======\n{out}",
        );
    }
}

fn check_non_json_output(edition: &str, expected_reports: usize) {
    let out = run_test(edition, None);
    let mut found_report = 0;
    for (line_nb, line) in out.lines().enumerate() {
        if line.starts_with('{') && serde_json::from_str::<serde_json::Value>(&line).is_ok() {
            panic!(
                "failed for {edition} edition: unexpected json at line {}: `{line}`\n\
                 ====== output ======\n{out}",
                line_nb + 1
            );
        }
        if line.starts_with("all doctests ran in")
            && line.contains("; merged doctests compilation took ")
        {
            found_report += 1;
        }
    }
    if found_report != expected_reports {
        panic!(
            "failed for {edition} edition: expected {expected_reports} doctest time `report`, \
             found {found_report}\n====== output ======\n{out}",
        );
    }
}

fn main() {
    // Only the merged doctests generate the "times report".
    check_json_output("2021", 0);
    check_json_output("2024", 1);

    // Only the merged doctests generate the "times report".
    check_non_json_output("2021", 0);
    check_non_json_output("2024", 1);
}
