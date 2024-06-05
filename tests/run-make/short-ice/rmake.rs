// Backtraces in internal compiler errors used to be unbearably long, spanning
// multiple hundreds of lines. A fix was pushed in #108938, and this test gathers
// varied metrics on level 1 and full-level backtraces to check that the output
// was shortened down to an appropriate length.
// See https://github.com/rust-lang/rust/issues/107910

use run_make_support::rustc;
use std::env;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut rust_test_1 = rustc().input("src/lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    env::set_var("RUST_BACKTRACE", "full");
    let mut rust_test_2 = rustc().input("src/lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    let rust_test_log_1 = rust_test_1.stderr_utf8().push_str(&rust_test_1.stdout_utf8()).as_str();
    let rust_test_log_2 = rust_test_2.stderr_utf8().push_str(&rust_test_2.stdout_utf8()).as_str();

    let rustc_query_count_full = count_lines_with(rust_test_log_2, "rustc_query_");

    assert!(
        rust_test_log_1.lines().count() < rust_test_log_2.lines().count()
            && count_lines_with(rust_test_log_2, "__rust_begin_short_backtrace")
                == count_lines_with(rust_test_log_2, "__rust_end_short_backtrace")
            && count_lines_with(rust_test_log_1, "rustc_query_") + 5 < rustc_query_count_full
            && rustc_query_count_full > 5
    );
}

fn count_lines_with(s: &str, search: &str) -> usize {
    s.lines().filter(|l| l.contains(search)).count()
}
