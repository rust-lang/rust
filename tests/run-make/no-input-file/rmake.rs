use run_make_support::{diff, rustc};

fn main() {
    let output = rustc().print("crate-name").run_fail_assert_exit_code(1);

    diff().expected_file("no-input-file.stderr").actual_text("output", output.stderr).run();
}
