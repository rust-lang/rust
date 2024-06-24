//@ edition: 2021
//@ run-fail
//@ check-run-results
//@ check-run-stdout-is-json-lines
//@ needs-unwind (for #[should_panic])
// ignore-tidy-linelength

//@ revisions: normal show-output
//@ compile-flags: --test
//@ run-flags: --test-threads=1 -Zunstable-options --format=json
//@ [show-output] run-flags: --show-output
//@ normalize-stdout-test: "(?<prefix>format-json.rs:)[0-9]+(?<suffix>:[0-9]+)" -> "${prefix}LL${suffix}"
//@ normalize-stdout-test: "(?<prefix>\"exec_time\": *)[0-9.]+" -> "${prefix}\"$$EXEC_TIME\""

// Check that passing `--format=json` to the test harness produces output that
// matches the snapshot, and is valid JSON-lines.

#[test]
fn a() {
    println!("print from successful test");
    // Should pass
}

#[test]
fn b() {
    assert!(false);
}

#[test]
#[should_panic]
fn c() {
    assert!(false);
}

#[test]
#[ignore = "msg"]
fn d() {
    assert!(false);
}
