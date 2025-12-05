// (This line has BOM so it's ignored by compiletest for directives)
//
//@ compile-flags: --json=diagnostic-short --error-format=json
//@ reference: input.byte-order-mark
//@ reference: input.crlf
// ignore-tidy-cr

// For easier verifying, the byte offsets in this file should match those
// in the json_bom_plus_crlf_multifile_aux.rs - given the actual fn is
// identical (just with a different, but equally sized name), the easiest way
// to do this is to ensure the two files are of equal size on disk.

// N.B., this file needs CRLF line endings. The .gitattributes file in
// this directory should enforce it.

fn main() {

    let s : String = 1;  // Error in the middle of line.
    //~^ ERROR mismatched types

    let s : String = 1
    ;  // Error before the newline.
    //~^^ ERROR mismatched types

    let s : String =
1;  // Error after the newline.
    //~^ ERROR mismatched types

    let s : String = (
    );  // Error spanning the newline.
    //~^^ ERROR mismatched types
}
