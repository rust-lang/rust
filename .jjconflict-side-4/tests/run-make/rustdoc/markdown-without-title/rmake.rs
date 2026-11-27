// When rustdoc gets a markdown file as input, we want to ensure that if the markdown is invalid,
// the output file won't be truncated in case this markdown is invalid.

//@ needs-target-std

use run_make_support::{path, rfs, rustdoc};

fn main() {
    let output_content = "output";
    let base_file_name = "input";

    let out_dir = path("out");
    rfs::create_dir(&out_dir);

    // We create the file that should be created by rustdoc and add some content
    // into it that we will check is still there once rustdoc failed.
    let output = out_dir.join(format!("{base_file_name}.html"));
    rfs::write(&output, output_content);

    // We create an "invalid" markdown file (ie no title).
    let md_file = format!("{base_file_name}.md");
    rfs::write(&md_file, "Markdown without a title");

    // We run the failing rustdoc.
    rustdoc()
        .input(&md_file)
        .out_dir(&out_dir)
        .run_fail()
        .assert_exit_code(1)
        .assert_stderr_contains(
            "error: invalid markdown file: no initial lines starting with `# ` or `%`",
        );

    // Shouldn't have changed.
    assert_eq!(rfs::read_to_string(&output), output_content);

    // We update the input markdown to make it valid for rustdoc.
    rfs::write(&md_file, "# a title\n\nMarkdown with a title");

    // We run rustdoc successfully.
    rustdoc().input(&md_file).out_dir(&out_dir).run();

    // Should have changed.
    assert_ne!(rfs::read_to_string(output), output_content);
}
