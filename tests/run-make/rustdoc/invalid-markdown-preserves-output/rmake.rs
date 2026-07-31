// Check that rejecting malformed standalone Markdown does not truncate an existing output file.

//@ needs-target-std

use run_make_support::{path, rfs, rustdoc};

const SENTINEL: &str = "existing output";

fn main() {
    let out_dir = path("out");
    rfs::create_dir(&out_dir);

    let output = out_dir.join("missing-title.html");
    rfs::write(&output, SENTINEL);

    rustdoc()
        .input("missing-title.md")
        .out_dir(&out_dir)
        .run_fail()
        .assert_exit_code(1)
        .assert_stderr_contains(
            "error: invalid markdown file: no initial lines starting with `# ` or `%`",
        );

    assert_eq!(rfs::read_to_string(output), SENTINEL);
}
