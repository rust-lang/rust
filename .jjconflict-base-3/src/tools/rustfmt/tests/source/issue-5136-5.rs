// rustfmt-file_lines: []
// Test that the space before the comment is not removed if the line is not
// contained in `--file-lines`.
// Note: It's important for the bug to repro that there is no newline at the
// end of the comment
fn f(){} // what