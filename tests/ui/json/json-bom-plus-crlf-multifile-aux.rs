// (This line has BOM so it's ignored by compiletest for directives)
//
//@ ignore-test Not a test. Used by other tests
// ignore-tidy-cr

// For easier verifying, the byte offsets in this file should match those
// in the json-bom-plus-crlf.rs - given the actual fn is identical (just with
// a different, but equally sized name), the easiest way to do this is to
// ensure the two files are of equal size on disk.
// Padding............................

// N.B., this file needs CRLF line endings. The .gitattributes file in
// this directory should enforce it.

pub fn test() {

    let s : String = 1;  // Error in the middle of line.

    let s : String = 1
    ;  // Error before the newline.

    let s : String =
1;  // Error after the newline.

    let s : String = (
    );  // Error spanning the newline.
}
