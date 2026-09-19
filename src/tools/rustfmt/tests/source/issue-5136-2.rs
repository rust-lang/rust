    use std;
// Test that whitespace at beginning of file is preserved when not in
// --file-lines range.
// This should prevent rustfmt from making any formatting changes at all:
// rustfmt-file_lines: []
