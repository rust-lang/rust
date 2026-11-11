use run_make_support::{diff, rustdoc};

fn main() {
    let invalid_print_request_help =
        rustdoc().arg("-Zunstable-options").arg("--print=xxx").run_fail().stderr_utf8();
    diff()
        .expected_file("invalid-print-request-help.err")
        .actual_text("invalid_print_request_help", &invalid_print_request_help)
        .run();
}
