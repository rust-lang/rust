//! Check that unstable print requests are omitted from help if compiler is in stable channel.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/138698>
use run_make_support::{diff, rustc, similar};

fn main() {
    let stable_invalid_print_request_help = rustc()
        .env("RUSTC_BOOTSTRAP", "-1")
        .cfg("force_stable")
        .print("xxx")
        .run_fail()
        .stderr_utf8();
    assert!(!stable_invalid_print_request_help.contains("all-target-specs-json"));
    diff()
        .expected_file("stable-invalid-print-request-help.err")
        .actual_text("stable_invalid_print_request_help", &stable_invalid_print_request_help)
        .run();

    let unstable_invalid_print_request_help = rustc().print("xxx").run_fail().stderr_utf8();
    assert!(unstable_invalid_print_request_help.contains("all-target-specs-json"));
    diff()
        .expected_file("unstable-invalid-print-request-help.err")
        .actual_text("unstable_invalid_print_request_help", &unstable_invalid_print_request_help)
        .run();

    let help_diff = similar::TextDiff::from_lines(
        &stable_invalid_print_request_help,
        &unstable_invalid_print_request_help,
    )
    .unified_diff()
    .to_string();
    diff().expected_file("help-diff.diff").actual_text("help_diff", help_diff).run();
}
