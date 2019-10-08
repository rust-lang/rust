// build-fail
// compile-flags: --json=diagnostic-short --error-format=json
// ignore-tidy-cr

// N.B., this file needs CRLF line endings. The .gitattributes file in
// this directory should enforce it.

fn main() {

    let s : String = 1;  // Error in the middle of line.

    let s : String = 1
    ;  // Error before the newline.

    let s : String =
1;  // Error after the newline.

    let s : String = (
    );  // Error spanning the newline.
}
