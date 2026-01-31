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

use run_make_support::CompletedProcess;

fn main() {
    // Run the same command twice with `RUST_BACKTRACE=1` and `RUST_BACKTRACE=full`.
    let configure_rustc = || {
        let mut rustc = run_make_support::rustc();
        rustc.input("src/lib.rs").arg("-Ztreat-err-as-bug=1");
        rustc
    };
    let rustc_bt_short = configure_rustc().set_backtrace_level("1").run_fail();
    let rustc_bt_full = configure_rustc().set_backtrace_level("full").run_fail();

    // Combine stderr and stdout for subsequent checks.
    let concat_stderr_stdout =
        |proc: &CompletedProcess| format!("{}\n{}", proc.stderr_utf8(), proc.stdout_utf8());
    let output_bt_short = &concat_stderr_stdout(&rustc_bt_short);
    let output_bt_full = &concat_stderr_stdout(&rustc_bt_full);

    // Count how many lines of output mention symbols or paths in
    // `rustc_query_system` or `rustc_query_impl`, which are the kinds of
    // stack frames we want to be omitting in short backtraces.
    let rustc_query_count_short = count_lines_with(output_bt_short, "rustc_query_");
    let rustc_query_count_full = count_lines_with(output_bt_full, "rustc_query_");

    // Dump both outputs in full to make debugging easier, especially on CI.
    // Use `--no-capture --force-rerun` to view output even when the test is passing.
    println!("=== BEGIN SHORT BACKTRACE ===\n{output_bt_short}\n=== END SHORT BACKTRACE === ");
    println!("=== BEGIN FULL BACKTRACE ===\n{output_bt_full}\n=== END FULL BACKTRACE === ");

    assert!(
        output_bt_short.lines().count() < output_bt_full.lines().count(),
        "Short backtrace should be shorter than full backtrace"
    );

    let n_begin = count_lines_with(output_bt_full, "__rust_begin_short_backtrace");
    let n_end = count_lines_with(output_bt_full, "__rust_end_short_backtrace");
    assert!(n_begin + n_end > 0, "Full backtrace should contain short-backtrace markers");
    assert_eq!(
        n_begin, n_end,
        "Full backtrace should contain equal numbers of begin and end markers"
    );

    assert!(
        rustc_query_count_short + 5 < rustc_query_count_full,
        "Short backtrace should have omitted more query plumbing lines \
        (actual: {rustc_query_count_short} vs {rustc_query_count_full})"
    );
}

fn count_lines_with(s: &str, search: &str) -> usize {
    s.lines().filter(|l| l.contains(search)).count()
}
