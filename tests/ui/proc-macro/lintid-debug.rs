//@ run-fail
//@ proc-macro: lint-id.rs

extern crate lint_id;

use std::process::ExitCode;

lint_id::print_lintid_to_stderr!();

fn main() -> ExitCode {
    ExitCode::FAILURE
}
