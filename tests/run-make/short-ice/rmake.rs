// Backtraces in internal compiler errors used to be unbearably long, spanning
// multiple hundreds of lines. A fix was pushed in #108938, and this test gathers
// varied metrics on level 1 and full-level backtraces to check that the output
// was shortened down to an appropriate length.
// See https://github.com/rust-lang/rust/issues/107910

//@ needs-target-std
//@ ignore-windows
// Reason: the assert_eq! on line 32 fails, as error output on Windows is different.

use run_make_support::rustc;

fn main() {
    let rust_test_1 =
        rustc().set_backtrace_level("1").input("src/lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    let rust_test_2 = rustc()
        .set_backtrace_level("full")
        .input("src/lib.rs")
        .arg("-Ztreat-err-as-bug=1")
        .run_fail();

    let mut rust_test_log_1 = rust_test_1.stderr_utf8();
    rust_test_log_1.push_str(&rust_test_1.stdout_utf8());
    let rust_test_log_1 = rust_test_log_1.as_str();

    let mut rust_test_log_2 = rust_test_2.stderr_utf8();
    rust_test_log_2.push_str(&rust_test_2.stdout_utf8());
    let rust_test_log_2 = rust_test_log_2.as_str();

    let rustc_query_count_full = count_lines_with(rust_test_log_2, "rustc_query_");

    assert!(rust_test_log_1.lines().count() < rust_test_log_2.lines().count());
    assert_eq!(
        count_lines_with(rust_test_log_2, "__rust_begin_short_backtrace"),
        count_lines_with(rust_test_log_2, "__rust_end_short_backtrace")
    );
    assert!(count_lines_with(rust_test_log_1, "rustc_query_") + 5 < rustc_query_count_full);
    assert!(rustc_query_count_full > 5);
}

fn count_lines_with(s: &str, search: &str) -> usize {
    s.lines().filter(|l| l.contains(search)).count()
}
