// Backtraces in internal compiler errors used to be unbearably long, spanning
// multiple hundreds of lines. A fix was pushed in #108938, and this test gathers
// varied metrics on level 1 and full-level backtraces to check that the output
// was shortened down to an appropriate length.
// See https://github.com/rust-lang/rust/issues/107910

//@ needs-target-std
//@ ignore-windows-msvc
//
// - FIXME(#143198): On `i686-pc-windows-msvc`: the assert_eq! on line 37 fails, almost seems like
//   it missing debug info? Haven't been able to reproduce locally, but it happens on CI.
// - FIXME(#143198): On `x86_64-pc-windows-msvc`: full backtrace sometimes do not contain matching
//   count of short backtrace markers (e.g. 5x end marker, but 3x start marker).

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

    assert!(
        rust_test_log_1.lines().count() < rust_test_log_2.lines().count(),
        "Short backtrace should be shorter than full backtrace.\nShort backtrace:\n\
        {rust_test_log_1}\nFull backtrace:\n{rust_test_log_2}"
    );
    assert_eq!(
        count_lines_with(rust_test_log_2, "__rust_begin_short_backtrace"),
        count_lines_with(rust_test_log_2, "__rust_end_short_backtrace"),
        "Full backtrace should contain the short backtrace markers.\nFull backtrace:\n\
        {rust_test_log_2}"
    );
    assert!(count_lines_with(rust_test_log_1, "rustc_query_") + 5 < rustc_query_count_full);
    assert!(rustc_query_count_full > 5);
}

fn count_lines_with(s: &str, search: &str) -> usize {
    s.lines().filter(|l| l.contains(search)).count()
}
